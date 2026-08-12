`timescale 1ns/1ps

// Synthesizable chunked fixed-point RMSNorm unit.
// Accepts CHUNKS rows of LANES x 32-bit fixed-point data, computes one shared
// reciprocal RMS over all CHUNKS*LANES values, then emits the normalized rows
// in the same order.  CHUNKS=1 preserves the old row-local behavior.
module rmsnorm #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32,
    parameter int CHUNKS = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta,
    output logic                    in_ready,
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    input  logic                    out_ready,
    output logic                    busy_o
);

    localparam int ROW_W = LANES * LANE_W;
    localparam int CHUNK_IDX_W = (CHUNKS <= 1) ? 1 : $clog2(CHUNKS);

    typedef enum logic [1:0] {
        ST_CAPTURE,
        ST_EMIT
    } state_e;

    state_e state_q;
    logic [ROW_W-1:0] row_buf [0:CHUNKS-1];
    logic [CHUNK_IDX_W-1:0] capture_idx_q;
    logic [CHUNK_IDX_W-1:0] emit_idx_q;
    logic [63:0] sum_sq_acc_q;
    logic [31:0] inv_rms_q15_q;
    logic [63:0] row_sum_sq;
    logic [63:0] total_sum_sq;
    logic [63:0] mean_square;
    logic [31:0] inv_rms_next;
    logic [ROW_W-1:0] normalized_row;
    logic unused_beta;

    assign unused_beta = ^beta;

    function automatic logic [15:0] get_lut_val(input logic [4:0] idx);
        case (idx)
            5'd0:  get_lut_val = 16'd32768;
            5'd1:  get_lut_val = 16'd32272;
            5'd2:  get_lut_val = 16'd31792;
            5'd3:  get_lut_val = 16'd31327;
            5'd4:  get_lut_val = 16'd30877;
            5'd5:  get_lut_val = 16'd30441;
            5'd6:  get_lut_val = 16'd30018;
            5'd7:  get_lut_val = 16'd29608;
            5'd8:  get_lut_val = 16'd29209;
            5'd9:  get_lut_val = 16'd28822;
            5'd10: get_lut_val = 16'd28445;
            5'd11: get_lut_val = 16'd28079;
            5'd12: get_lut_val = 16'd27722;
            5'd13: get_lut_val = 16'd27375;
            5'd14: get_lut_val = 16'd27037;
            5'd15: get_lut_val = 16'd26708;
            5'd16: get_lut_val = 16'd26387;
            5'd17: get_lut_val = 16'd26074;
            5'd18: get_lut_val = 16'd25769;
            5'd19: get_lut_val = 16'd25471;
            5'd20: get_lut_val = 16'd25181;
            5'd21: get_lut_val = 16'd24898;
            5'd22: get_lut_val = 16'd24621;
            5'd23: get_lut_val = 16'd24351;
            5'd24: get_lut_val = 16'd24087;
            5'd25: get_lut_val = 16'd23829;
            5'd26: get_lut_val = 16'd23577;
            5'd27: get_lut_val = 16'd23331;
            5'd28: get_lut_val = 16'd23090;
            5'd29: get_lut_val = 16'd22854;
            5'd30: get_lut_val = 16'd22624;
            5'd31: get_lut_val = 16'd22398;
            default: get_lut_val = 16'd32768;
        endcase
    endfunction

    function automatic integer get_msb64(input logic [63:0] val);
        integer pos;
        begin
            pos = 0;
            for (integer k = 0; k < 64; k++) begin
                if (val[k])
                    pos = k;
            end
            get_msb64 = pos;
        end
    endfunction

    function automatic logic [31:0] compute_inv_sqrt(input logic [63:0] ms_val);
        integer msb;
        logic [63:0] ms_norm;
        logic [4:0]  idx;
        logic [31:0] raw_lut;
        logic [31:0] res;
        logic [39:0] res_q8_8_adjusted;
        integer half_msb;
        begin
            if (ms_val == 64'd0) begin
                compute_inv_sqrt = 32'd32767;
            end else begin
                msb = get_msb64(ms_val);
                if (msb < 5)
                    ms_norm = ms_val << (5 - msb);
                else
                    ms_norm = ms_val >> (msb - 5);
                idx = ms_norm[4:0];
                raw_lut = {16'b0, get_lut_val(idx)};

                half_msb = msb / 2;
                res = raw_lut >> half_msb;
                if (msb % 2 != 0)
                    res = (res * 32'd23170) >> 15;
                // mean_square is computed from Q8.8 lanes, so it is Q16.16.
                // The reciprocal square-root therefore needs an extra 2**8
                // factor to return a Q1.15 gain for the original real values.
                res_q8_8_adjusted = {8'd0, res} << 8;
                compute_inv_sqrt = (res_q8_8_adjusted > 40'd2147483647)
                    ? 32'd2147483647
                    : res_q8_8_adjusted[31:0];
            end
        end
    endfunction

    always_comb begin
        row_sum_sq = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [LANE_W-1:0] x_lane;
            logic signed [63:0] sq;
            x_lane = $signed(x_in[lane*LANE_W +: LANE_W]);
            sq = 64'(x_lane) * 64'(x_lane);
            row_sum_sq = row_sum_sq + sq[63:0];
        end

        total_sum_sq = sum_sq_acc_q + row_sum_sq;
        mean_square = total_sum_sq / (CHUNKS * LANES);
        inv_rms_next = compute_inv_sqrt(mean_square);
    end

    always_comb begin
        normalized_row = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [LANE_W-1:0] x_lane;
            logic signed [LANE_W-1:0] g_lane;
            logic signed [63:0] prod;
            logic signed [63:0] final_lane;

            x_lane = $signed(row_buf[emit_idx_q][lane*LANE_W +: LANE_W]);
            g_lane = $signed(gamma[lane*LANE_W +: LANE_W]);
            if (g_lane == 32'd0)
                g_lane = 32'sd128;

            // x_lane is carried through VXM as Q8.8, inv_rms_q15_q is Q1.15,
            // and gamma is Q1.7.  Keep the output in Q8.8 by removing both
            // the reciprocal RMS fractional bits and the gamma fractional bits.
            prod = 64'(x_lane) * 64'($signed(inv_rms_q15_q)) * 64'(g_lane);
            final_lane = prod >>> 22;
            normalized_row[lane*LANE_W +: LANE_W] = LANE_W'(final_lane);
        end
    end

    assign in_ready = (state_q == ST_CAPTURE);
    assign done_o   = (state_q == ST_EMIT);
    assign busy_o   = (state_q != ST_CAPTURE);
    assign y_out    = normalized_row;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q       <= ST_CAPTURE;
            capture_idx_q <= '0;
            emit_idx_q    <= '0;
            sum_sq_acc_q  <= '0;
            inv_rms_q15_q <= '0;
            for (int chunk = 0; chunk < CHUNKS; chunk++) begin
                row_buf[chunk] <= '0;
            end
        end else begin
            unique case (state_q)
                ST_CAPTURE: begin
                    if (start_i) begin
                        row_buf[capture_idx_q] <= x_in;
                        if (capture_idx_q == CHUNK_IDX_W'(CHUNKS-1)) begin
                            inv_rms_q15_q <= inv_rms_next;
                            capture_idx_q <= '0;
                            emit_idx_q    <= '0;
                            sum_sq_acc_q  <= '0;
                            state_q       <= ST_EMIT;
                        end else begin
                            sum_sq_acc_q  <= total_sum_sq;
                            capture_idx_q <= capture_idx_q + 1'b1;
                        end
                    end
                end

                ST_EMIT: begin
                    if (out_ready) begin
                        if (emit_idx_q == CHUNK_IDX_W'(CHUNKS-1)) begin
                            emit_idx_q <= '0;
                            state_q    <= ST_CAPTURE;
                        end else begin
                            emit_idx_q <= emit_idx_q + 1'b1;
                        end
                    end
                end

                default: begin
                    state_q <= ST_CAPTURE;
                end
            endcase
        end
    end

endmodule
