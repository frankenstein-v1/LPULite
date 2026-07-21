`timescale 1ns/1ns

`ifndef SYNTHESIS

// // FP32 row layernorm for VXM.
// FP32 row RMSNorm for VXM.
//
// Lanes are raw IEEE-754 FP32 bit patterns. This keeps the existing module
// // name because VXM already instantiates lut_layernorm, but the behavior is now
// name because VXM now instantiates lut_rmsnorm, and the behavior is
// floating-point RMSNorm.
module lut_rmsnorm #(
    parameter int LANES      = 8,
    parameter int LANE_W     = 32,
    parameter int LUT_FRAC_W = 16
) (
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta, // kept for interface compatibility, unused in math
    output logic [LANES*LANE_W-1:0] y_out
);

    localparam real EPSILON = 0.00001;

    function automatic logic [63:0] f32_to_f64_bits(logic [31:0] f32);
        logic sign;
        logic [7:0] exp32;
        logic [22:0] mant32;
        logic [10:0] exp64;
        logic [51:0] mant64;
        int biased_exp;
        begin
            sign = f32[31];
            exp32 = f32[30:23];
            mant32 = f32[22:0];
            if (exp32 == 8'h00) begin
                exp64 = 11'h000;
                mant64 = {mant32, 29'b0};
            end else if (exp32 == 8'hFF) begin
                exp64 = 11'h7FF;
                mant64 = {mant32, 29'b0};
            end else begin
                biased_exp = int'(exp32) - 127 + 1023;
                exp64 = biased_exp[10:0];
                mant64 = {mant32, 29'b0};
            end
            f32_to_f64_bits = {sign, exp64, mant64};
        end
    endfunction

    function automatic logic [31:0] f64_to_f32_bits(logic [63:0] f64);
        logic sign;
        logic [10:0] exp64;
        logic [51:0] mant64;
        logic [7:0] exp32;
        logic [22:0] mant32;
        int biased_exp;
        begin
            sign = f64[63];
            exp64 = f64[62:52];
            mant64 = f64[51:0];
            if (exp64 == 11'h000) begin
                exp32 = 8'h00;
                mant32 = mant64[51:29];
            end else if (exp64 == 11'h7FF) begin
                exp32 = 8'hFF;
                mant32 = mant64[51:29];
            end else begin
                biased_exp = int'(exp64) - 1023 + 127;
                if (biased_exp <= 0) begin
                    exp32 = 8'h00;
                    mant32 = 23'h0;
                end else if (biased_exp >= 255) begin
                    exp32 = 8'hFF;
                    mant32 = 23'h0;
                end else begin
                    exp32 = biased_exp[7:0];
                    mant32 = mant64[51:29];
                end
            end
            f64_to_f32_bits = {sign, exp32, mant32};
        end
    endfunction

    // Look-up table to get the value of 1/sqrroot(RMS^2)
    function automatic real lookup_inv_sqrt(real val);
        begin
            lookup_inv_sqrt = 1.0 / $sqrt(val + EPSILON);
        end
    endfunction

    // function automatic logic [LANES*LANE_W-1:0] compute_layernorm(
    function automatic logic [LANES*LANE_W-1:0] compute_rmsnorm(
        input logic [LANES*LANE_W-1:0] x_in_val,
        input logic [LANES*LANE_W-1:0] gamma_val
    );
        real x_lane [0:LANES-1];
        real gamma_lane [0:LANES-1];
        real rms_sq;
        real sum_sq;
        real inv_rms;
        real y_lane;
        logic [31:0] x_bits;
        logic [31:0] gamma_bits;
        logic [LANES*LANE_W-1:0] result;
        begin
            sum_sq = 0.0;
            result = '0;

            for (int i = 0; i < LANES; i++) begin
                x_bits = x_in_val[i*LANE_W +: LANE_W];
                gamma_bits = gamma_val[i*LANE_W +: LANE_W];

                x_lane[i] = $bitstoreal(f32_to_f64_bits(x_bits));
                gamma_lane[i] = $bitstoreal(f32_to_f64_bits(gamma_bits));
                sum_sq = sum_sq + (x_lane[i] * x_lane[i]);
            end

            // d value is the configured lane count.
            rms_sq = sum_sq / LANES;

            inv_rms = lookup_inv_sqrt(rms_sq);

            for (int i = 0; i < LANES; i++) begin
                y_lane = x_lane[i] * inv_rms * gamma_lane[i];
                result[i*LANE_W +: LANE_W] = f64_to_f32_bits($realtobits(y_lane));
            end
            
            // compute_layernorm = result;
            compute_rmsnorm = result;
        end
    endfunction

    always @(x_in or gamma) begin
        // y_out = compute_layernorm(x_in, gamma, beta);
        y_out = compute_rmsnorm(x_in, gamma);
    end

endmodule
`endif
