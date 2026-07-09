`timescale 1ns/1ns

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

    always_comb begin
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

        mean = 0.0;
        variance = 0.0;
        y_out = '0;

        for (int i = 0; i < LANES; i++) begin
            x_bits = x_in[i*LANE_W +: LANE_W];
            gamma_bits = gamma[i*LANE_W +: LANE_W];
            beta_bits = beta[i*LANE_W +: LANE_W];

            x_lane[i] = $bitstoshortreal(x_bits);
            gamma_lane[i] = $bitstoshortreal(gamma_bits);
            beta_lane[i] = $bitstoshortreal(beta_bits);
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
            y_out[i*LANE_W +: LANE_W] = $shortrealtobits(shortreal'(y_lane));
        end
    end

endmodule
