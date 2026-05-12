module quant #(
    parameter int SHIFT = 16
) (
    input  logic               clk,
    input  logic               rst,
    input  logic signed [127:0] x_input,
    output logic signed [31:0]  q_row_out
);

//TODO: @Saksham fixed point quantization 
// q = clamp(round(255 * p), 0, 255)


endmodule
