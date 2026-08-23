// ==============================================================================
// Original Mathematical Implementation Credit: SuryaHead
// Repository: https://github.com/SurjaHead/softmax-in-hardware
// Implementation based on I-BERT paper integer arithmetic.
// Modified for LPULite Integration.
// ==============================================================================
`default_nettype none
`timescale 1ns/1ns

module exp #(
  parameter int DW = 32
)(
  input  logic         clk,
  input  logic         rst,
  input  logic signed [DW-1:0] q,
  output logic signed [DW-1:0] q_out
);
    // Constants
    localparam int LN2 = 177;      // ln(2) * 256 ≈ 177
    localparam int A = 92;         // Scaled coefficient ≈ 0.3585 * 256
    localparam int B = 346;        // Shift term ≈ 1.353 * 256
    localparam int C = 88;         // Constant term ≈ 0.344 * 256

    // Decompose q into z and p: q = z * LN2 + p
    logic signed [DW-1:0] z;
    assign z = -q / LN2;  // Integer part

    logic signed [DW-1:0] p;
    assign p = q + (z * LN2);  // Fractional part

    // Polynomial approximation: L(p) = 0.3585 * (p + 1.353)^2 + 0.344
    logic signed [DW-1:0] t;
    assign t = p + B;  // t = p + 1.353 * 256

    logic signed [2*DW-1:0] t2;
    assign t2 = t * t;

    logic signed [3*DW-1:0] q_L;
    assign q_L = ((A * t2) >>> 16) + C;  // re-quantize to keep scale of 256

    assign q_out = q_L >>> z;   //(z_signed >= 0) ? (q_L << z_signed) : (q_L >> (-z_signed));



endmodule
