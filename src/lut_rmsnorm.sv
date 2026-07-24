`timescale 1ns/1ps

// Synthesizable 32-bit signed fixed-point RMSNorm with reciprocal square-root LUT.
module lut_rmsnorm #(
    parameter int LANES      = 8,
    parameter int LANE_W     = 32,
    parameter int LUT_FRAC_W = 16
) (
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta, // Interface compatibility
    output logic [LANES*LANE_W-1:0] y_out
);

    // Precomputed Q1.15 reciprocal sqrt table for normalized range [1.0, 2.0)
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

    // Function to calculate 5-bit MSB position of a 64-bit unsigned integer
    function automatic integer get_msb64(input logic [63:0] val);
        integer pos;
        begin
            pos = 0;
            for (integer k = 0; k < 64; k++) begin
                if (val[k]) pos = k;
            end
            get_msb64 = pos;
        end
    endfunction

    // Function to compute Q1.15 inv_sqrt of a 64-bit mean square value
    function automatic logic [31:0] compute_inv_sqrt(input logic [63:0] ms_val);
        integer msb;
        logic [63:0] ms_norm;
        logic [4:0]  idx;
        logic [31:0] raw_lut;
        logic [31:0] res;
        integer half_msb;
        begin
            if (ms_val == 64'd0) begin
                compute_inv_sqrt = 32'd32767;
            end else begin
                msb = get_msb64(ms_val);
                if (msb < 5) begin
                    ms_norm = ms_val << (5 - msb);
                end else begin
                    ms_norm = ms_val >> (msb - 5);
                end
                idx = ms_norm & 64'h1F;
                raw_lut = {16'b0, get_lut_val(idx)};

                half_msb = msb / 2;
                res = raw_lut >> half_msb;
                if (msb % 2 != 0) begin
                    // 1/sqrt(2) approx 23170 in Q1.15
                    res = (res * 32'd23170) >> 15;
                end
                compute_inv_sqrt = res;
            end
        end
    endfunction

    // Combinatorial calculation across lanes
    always_comb begin
        logic signed [63:0] sum_sq;
        logic [63:0]        ms_val;
        logic [31:0]        inv_rms_q15;

        sum_sq = '0;
        for (int i = 0; i < LANES; i++) begin
            logic signed [LANE_W-1:0] x_lane;
            logic signed [63:0] sq;
            x_lane = $signed(x_in[i*LANE_W +: LANE_W]);
            sq = 64'(x_lane) * 64'(x_lane);
            sum_sq = sum_sq + sq;
        end

        // Divide sum_sq by LANES (LANES=8 -> shift right by 3)
        ms_val = sum_sq >> 3;

        inv_rms_q15 = compute_inv_sqrt(ms_val);

        for (int i = 0; i < LANES; i++) begin
            logic signed [LANE_W-1:0] x_lane;
            logic signed [LANE_W-1:0] g_lane;
            logic signed [63:0] prod;
            logic signed [63:0] final_lane;

            x_lane = $signed(x_in[i*LANE_W +: LANE_W]);
            g_lane = $signed(gamma[i*LANE_W +: LANE_W]);

            if (g_lane == 32'd0) begin
                g_lane = 32'sd128; // Default Q1.7 identity gain of 1.0
            end

            // Product: x_lane * inv_rms_q15 * g_lane
            // inv_rms_q15 is Q1.15, g_lane is Q1.7 (scaled by 128)
            // Shift right by 15 keeps the output scaled by g_lane
            prod = 64'(x_lane) * 64'($signed(inv_rms_q15)) * 64'(g_lane);
            final_lane = prod >>> 15;

            y_out[i*LANE_W +: LANE_W] = LANE_W'(final_lane);
        end
    end

endmodule
