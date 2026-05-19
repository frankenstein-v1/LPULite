module quant #(
    parameter int SHIFT = 16,
    parameter int LANES  = 4,
    parameter int LANE_W = 32
) (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               in_valid,
    input  logic signed [LANES*LANE_W-1:0] x_input,
    
    output logic               out_valid,
    output logic signed [LANES*LANE_W-1:0] q_row_out // Note: Kept width as LANES*LANE_W to match VXM pipeline
);

// ==============================================================================
// TODO: @Saksham - Implement fixed point quantization here!
// 
// This module sits in the VXM datapath between the Row Collector and the Softmax.
// You need to quantize the 128-bit `x_input` (4 lanes of 32-bits) and output it 
// on `q_row_out`.
// 
// Make sure to drive `out_valid` high when your `q_row_out` data is ready.
// If your logic is combinational, simply: assign out_valid = in_valid;
//
// Math reference: q = clamp(round(255 * p), 0, 255)
// ==============================================================================

    // Dummy pass-through for now so the pipeline compiles
    assign out_valid = in_valid;
    assign q_row_out = x_input;

endmodule
