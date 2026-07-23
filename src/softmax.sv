`timescale 1ns/1ns

module softmax #(
    parameter int LANES      = 8,
    parameter int LANE_W     = 32,
    parameter int MAX_CHUNKS = 64,
    parameter logic signed [7:0] PROB_SCALE = -8'sd7
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    in_valid,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic signed [7:0]       x_scale_i,
    output logic                    in_ready,

    output logic                    out_valid,
    output logic [LANES*LANE_W-1:0] y_out,
    output logic signed [7:0]       y_scale_o,
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
    logic signed [7:0] scale_buf [0:MAX_CHUNKS-1];
    logic [IDX_W-1:0] write_idx;
    logic [IDX_W-1:0] read_idx;
    logic signed [LANE_W-1:0] global_max;
    logic [31:0]      global_sum;
    logic             active_q;

    logic [ROW_W-1:0] active_row;
    logic signed [7:0] active_scale;
    logic signed [LANE_W-1:0] aligned_row [0:LANES-1];
    logic signed [LANE_W-1:0] row_max;
    logic [31:0]      exp_row [0:LANES-1];
    logic signed [LANE_W-1:0] exp_delta_q [0:LANES-1];
    logic [31:0]      chunk_sum;
    logic [7:0]       prob_u8 [0:LANES-1];

    function automatic logic signed [LANE_W-1:0] align_int32_to_q8_8(
        input logic signed [LANE_W-1:0] raw_value,
        input logic signed [7:0]        row_scale
    );
        integer shift_amount;
        longint signed scaled_value;
        begin
            shift_amount = $signed(row_scale) + 8;
            scaled_value = raw_value;

            if (shift_amount >= 0) begin
                if (shift_amount > 30)
                    scaled_value = raw_value[LANE_W-1] ? -64'sh8000_0000 : 64'sh7fff_ffff;
                else
                    scaled_value = scaled_value <<< shift_amount;
            end else if (-shift_amount > 62) begin
                scaled_value = raw_value[LANE_W-1] ? -64'sd1 : 64'sd0;
            end else begin
                scaled_value = scaled_value >>> (-shift_amount);
            end

            if (scaled_value > 64'sh7fff_ffff)
                align_int32_to_q8_8 = 32'sh7fff_ffff;
            else if (scaled_value < -64'sh8000_0000)
                align_int32_to_q8_8 = 32'sh8000_0000;
            else
                align_int32_to_q8_8 = scaled_value[LANE_W-1:0];
        end
    endfunction

    function automatic logic [7:0] softmax_prob_u8(
        input logic [31:0] exp_value,
        input logic [31:0] sum_value
    );
        longint unsigned numerator;
        longint unsigned quotient;
        begin
            if (sum_value == 32'd0) begin
                softmax_prob_u8 = 8'd0;
            end else begin
                numerator = ({32'd0, exp_value} << (-PROB_SCALE)) + {33'd0, sum_value[31:1]};
                quotient = numerator / {32'd0, sum_value};
                softmax_prob_u8 = (quotient > 64'd255) ? 8'd255 : quotient[7:0];
            end
        end
    endfunction

    always @* begin
        active_row = (state_q == ST_CAPTURE) ? x_in : score_buf[read_idx];
        active_scale = (state_q == ST_CAPTURE) ? x_scale_i : scale_buf[read_idx];

        for (int lane = 0; lane < LANES; lane++) begin
            aligned_row[lane] = align_int32_to_q8_8(
                $signed(active_row[lane*LANE_W +: LANE_W]),
                active_scale
            );
        end

        row_max = aligned_row[0];
        for (int lane = 1; lane < LANES; lane++) begin
            if (aligned_row[lane] > row_max)
                row_max = aligned_row[lane];
        end

        chunk_sum = 32'd0;
        for (int lane = 0; lane < LANES; lane++) begin
            exp_delta_q[lane] = aligned_row[lane] - global_max;
            chunk_sum = chunk_sum + exp_row[lane];
            prob_u8[lane] = softmax_prob_u8(exp_row[lane], global_sum);
            y_out[lane*LANE_W +: LANE_W] = {{(LANE_W-8){1'b0}}, prob_u8[lane]};
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
            global_max <= 32'sh8000_0000;
            global_sum <= 32'd0;
            active_q   <= 1'b0;
        end else begin
            unique case (state_q)
                ST_CAPTURE: begin
                    if (in_valid && in_ready) begin
                        score_buf[write_idx] <= x_in;
                        scale_buf[write_idx] <= x_scale_i;

                        if (!active_q || (row_max > global_max))
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
                            global_max <= 32'sh8000_0000;
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
    assign y_scale_o = PROB_SCALE;
    assign busy_o = active_q || (state_q != ST_CAPTURE);

endmodule
