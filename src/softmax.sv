`timescale 1ns/1ns

module softmax #(
    parameter int LANES      = 8,
    parameter int LANE_W     = 32,
    parameter int MAX_CHUNKS = 64
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    in_valid,
    input  logic [LANES*LANE_W-1:0] x_in,
    output logic                    in_ready,

    output logic                    out_valid,
    output logic                    out_mode_fp,
    output logic [LANES*LANE_W-1:0] y_out,
    input  logic                    out_ready,

    output logic                    busy_o
);

    localparam int ROW_W = LANES * LANE_W;
    localparam int IDX_W = (MAX_CHUNKS <= 1) ? 1 : $clog2(MAX_CHUNKS);

    typedef enum logic [1:0] {
        ST_CAPTURE,
        ST_SUM,
        ST_NORM
    } state_e;

    state_e state_q;

    logic [ROW_W-1:0] score_buf [0:MAX_CHUNKS-1];
    logic [IDX_W-1:0] write_idx;
    logic [IDX_W-1:0] read_idx;
    logic [31:0]      global_max;
    logic [31:0]      global_sum;
    logic             active_q;

    logic [ROW_W-1:0] active_row;
    logic [31:0]      row_max;
    logic [31:0]      exp_row [0:LANES-1];
    logic signed [LANE_W-1:0] exp_delta_q [0:LANES-1];
    logic [31:0]      chunk_sum;
    logic [31:0]      prob_fixed [0:LANES-1];

    function automatic logic fp32_gt(
        input logic [31:0] a,
        input logic [31:0] b
    );
        logic sign_a;
        logic sign_b;
        begin
            sign_a = a[31];
            sign_b = b[31];

            if (a[30:0] == 31'd0 && b[30:0] == 31'd0)
                fp32_gt = 1'b0;
            else if (sign_a != sign_b)
                fp32_gt = sign_b;
            else if (!sign_a)
                fp32_gt = a[30:0] > b[30:0];
            else
                fp32_gt = a[30:0] < b[30:0];
        end
    endfunction

    function automatic logic signed [LANE_W-1:0] fp32_to_q8_8(
        input logic [31:0] fp_bits
    );
        logic        sign_bit;
        logic [7:0]  exp_bits;
        logic [22:0] frac_bits;
        logic [23:0] significand;
        integer      exp_unbiased;
        integer      shift_amount;
        longint signed scaled_value;
        begin
            sign_bit = fp_bits[31];
            exp_bits = fp_bits[30:23];
            frac_bits = fp_bits[22:0];

            if ((exp_bits == 8'h00) && (frac_bits == 23'd0)) begin
                fp32_to_q8_8 = '0;
            end else if (exp_bits == 8'hff) begin
                fp32_to_q8_8 = sign_bit ? 32'sh8000_0000 : 32'sh7fff_ffff;
            end else begin
                if (exp_bits == 8'h00) begin
                    significand = {1'b0, frac_bits};
                    exp_unbiased = -126;
                end else begin
                    significand = {1'b1, frac_bits};
                    exp_unbiased = exp_bits - 127;
                end

                shift_amount = exp_unbiased - 23 + 8;
                scaled_value = $signed({1'b0, significand});

                if (shift_amount >= 0) begin
                    if (shift_amount > 30)
                        scaled_value = 64'sh7fff_ffff;
                    else
                        scaled_value = scaled_value <<< shift_amount;
                end else if (-shift_amount > 62) begin
                    scaled_value = 64'sd0;
                end else begin
                    scaled_value = scaled_value >>> (-shift_amount);
                end

                if (sign_bit)
                    scaled_value = -scaled_value;

                if (scaled_value > 64'sh7fff_ffff)
                    fp32_to_q8_8 = 32'sh7fff_ffff;
                else if (scaled_value < -64'sh8000_0000)
                    fp32_to_q8_8 = 32'sh8000_0000;
                else
                    fp32_to_q8_8 = scaled_value[LANE_W-1:0];
            end
        end
    endfunction

    function automatic logic [31:0] uq8_8_to_fp32(
        input logic [LANE_W-1:0] fixed_value
    );
        logic [31:0] normalized;
        logic [7:0]  exponent_bits;
        integer      msb_idx;
        begin
            if (fixed_value == '0) begin
                uq8_8_to_fp32 = 32'h0000_0000;
            end else begin
                msb_idx = 0;
                for (int idx = 0; idx < LANE_W; idx++) begin
                    if (((fixed_value >> idx) & 1'b1) != 1'b0)
                        msb_idx = idx;
                end

                exponent_bits = msb_idx + 8'd119;
                if (msb_idx <= 23)
                    normalized = fixed_value << (23 - msb_idx);
                else
                    normalized = fixed_value >> (msb_idx - 23);
                uq8_8_to_fp32 = {1'b0, exponent_bits, normalized[22:0]};
            end
        end
    endfunction

    function automatic logic [31:0] softmax_prob_q8_8(
        input logic [31:0] exp_value,
        input logic [31:0] sum_value
    );
        longint unsigned numerator;
        longint unsigned quotient;
        begin
            if (sum_value == 32'd0) begin
                softmax_prob_q8_8 = 32'd0;
            end else begin
                numerator = ({32'd0, exp_value} << 8) + {33'd0, sum_value[31:1]};
                quotient = numerator / {32'd0, sum_value};
                softmax_prob_q8_8 = (quotient > 64'd256) ? 32'd256 : quotient[31:0];
            end
        end
    endfunction

    always_comb begin
        active_row = (state_q == ST_CAPTURE) ? x_in : score_buf[read_idx];

        row_max = active_row[0 +: LANE_W];
        for (int lane = 1; lane < LANES; lane++) begin
            if (fp32_gt(active_row[lane*LANE_W +: LANE_W], row_max))
                row_max = active_row[lane*LANE_W +: LANE_W];
        end

        chunk_sum = 32'd0;
        for (int lane = 0; lane < LANES; lane++) begin
            exp_delta_q[lane] = fp32_to_q8_8(active_row[lane*LANE_W +: LANE_W])
                              - fp32_to_q8_8(global_max);
            chunk_sum = chunk_sum + exp_row[lane];
            prob_fixed[lane] = softmax_prob_q8_8(exp_row[lane], global_sum);
            y_out[lane*LANE_W +: LANE_W] = uq8_8_to_fp32(prob_fixed[lane]);
        end
    end

    generate
        genvar lane;
        for (lane = 0; lane < LANES; lane++) begin : gen_exp
            lut_softmax_exp #(.DW(LANE_W)) exp_inst (
                .clk(clk),
                .rst(~rst_n),
                .q(exp_delta_q[lane]),
                .q_out(exp_row[lane])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q    <= ST_CAPTURE;
            write_idx  <= '0;
            read_idx   <= '0;
            global_max <= 32'hff80_0000;
            global_sum <= 32'd0;
            active_q   <= 1'b0;
        end else begin
            unique case (state_q)
                ST_CAPTURE: begin
                    if (in_valid && in_ready) begin
                        score_buf[write_idx] <= x_in;

                        if (!active_q || fp32_gt(row_max, global_max))
                            global_max <= row_max;

                        active_q <= 1'b1;

                        if (write_idx == IDX_W'(MAX_CHUNKS-1)) begin
                            write_idx  <= '0;
                            read_idx   <= '0;
                            global_sum <= 32'd0;
                            state_q    <= ST_SUM;
                        end else begin
                            write_idx <= write_idx + 1'b1;
                        end
                    end
                end

                ST_SUM: begin
                    global_sum <= global_sum + chunk_sum;
                    if (read_idx == IDX_W'(MAX_CHUNKS-1)) begin
                        read_idx <= '0;
                        state_q  <= ST_NORM;
                    end else begin
                        read_idx <= read_idx + 1'b1;
                    end
                end

                ST_NORM: begin
                    if (out_ready) begin
                        if (read_idx == IDX_W'(MAX_CHUNKS-1)) begin
                            read_idx   <= '0;
                            write_idx  <= '0;
                            global_max <= 32'hff80_0000;
                            global_sum <= 32'd0;
                            active_q   <= 1'b0;
                            state_q    <= ST_CAPTURE;
                        end else begin
                            read_idx <= read_idx + 1'b1;
                        end
                    end
                end

                default: begin
                    state_q <= ST_CAPTURE;
                end
            endcase
        end
    end

    assign in_ready = (state_q == ST_CAPTURE);
    assign out_valid = (state_q == ST_NORM);
    assign out_mode_fp = 1'b1;
    assign busy_o = active_q || (state_q != ST_CAPTURE);

endmodule
