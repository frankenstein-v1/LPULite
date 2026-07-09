`timescale 1ns/1ns

`include "cvfpu_compat.svh"

// FP32 row layernorm for VXM.
//
// Lanes are raw IEEE-754 FP32 bit patterns. This keeps the existing module
// name because VXM already instantiates lut_layernorm, but the behavior is now
// floating-point rather than fixed-point LUT math.
module lut_layernorm #(
    parameter int LANES      = 4,
    parameter int LANE_W     = 32,
    parameter int LUT_FRAC_W = 16
) (
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta,
    output logic [LANES*LANE_W-1:0] y_out
);

    localparam real EPSILON = 0.00001;

    function automatic logic [LANES*LANE_W-1:0] compute_layernorm(
        input logic [LANES*LANE_W-1:0] x_in_val,
        input logic [LANES*LANE_W-1:0] gamma_val,
        input logic [LANES*LANE_W-1:0] beta_val
    );
        real x_lane [0:LANES-1];
        real gamma_lane [0:LANES-1];
        real beta_lane [0:LANES-1];
        real mean;
        real variance;
        real inv_std;
        real centered;
        real y_lane;
        logic [31:0] x_bits;
        logic [31:0] gamma_bits;
        logic [31:0] beta_bits;
        logic [LANES*LANE_W-1:0] result;
        begin
            mean = 0.0;
            variance = 0.0;
            result = '0;

            for (int i = 0; i < LANES; i++) begin
                x_bits = x_in_val[i*LANE_W +: LANE_W];
                gamma_bits = gamma_val[i*LANE_W +: LANE_W];
                beta_bits = beta_val[i*LANE_W +: LANE_W];

                x_lane[i] = $bitstoreal(f32_to_f64_bits(x_bits));
                gamma_lane[i] = $bitstoreal(f32_to_f64_bits(gamma_bits));
                beta_lane[i] = $bitstoreal(f32_to_f64_bits(beta_bits));
                mean = mean + x_lane[i];
            end

            mean = mean / LANES;

            for (int i = 0; i < LANES; i++) begin
                centered = x_lane[i] - mean;
                variance = variance + (centered * centered);
            end

            variance = variance / LANES;
            inv_std = 1.0 / $sqrt(variance + EPSILON);

            for (int i = 0; i < LANES; i++) begin
                centered = x_lane[i] - mean;
                y_lane = (centered * inv_std * gamma_lane[i]) + beta_lane[i];
                result[i*LANE_W +: LANE_W] = f64_to_f32_bits($realtobits(y_lane));
            end
            
            compute_layernorm = result;
        end
    endfunction

    always @(x_in or gamma or beta) begin
        y_out = compute_layernorm(x_in, gamma, beta);
    end

endmodule
