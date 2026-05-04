`timescale 1ns/1ps

module mxm_2x2_cocotb_top (
    input  logic              clk,
    input  logic              rst,
    input  logic              mxm_clear,
    input  logic              mxm_start,
    input  logic signed [7:0] input0,
    input  logic signed [7:0] input1,
    input  logic              wght_load0,
    input  logic              wght_load1,
    input  logic signed [7:0] wght_val0,
    input  logic signed [7:0] wght_val1,
    output logic signed [31:0] c00,
    output logic signed [31:0] c01,
    output logic signed [31:0] c10,
    output logic signed [31:0] c11
);

    logic signed [7:0] mxm_input_in [1:0];
    logic              wght_load  [1:0];
    logic signed [7:0] wght_val   [1:0];
    logic signed [31:0] mxm_out   [1:0][1:0];

    assign mxm_input_in[0] = input0;
    assign mxm_input_in[1] = input1;
    assign wght_load[0]  = wght_load0;
    assign wght_load[1]  = wght_load1;
    assign wght_val[0]   = wght_val0;
    assign wght_val[1]   = wght_val1;

    assign c00 = mxm_out[0][0];
    assign c01 = mxm_out[0][1];
    assign c10 = mxm_out[1][0];
    assign c11 = mxm_out[1][1];

    mxm #(
        .mxm_size(2)
    ) dut (
        .clk(clk),
        .rst(rst),
        .mxm_clear(mxm_clear),
        .mxm_start(mxm_start),
        .mxm_input_in(mxm_input_in),
        .wght_load(wght_load),
        .wght_val(wght_val),
        .mxm_out(mxm_out)
    );

`ifdef WAVEFORM
    initial begin
        $dumpfile("mxm_2x2_cocotb.vcd");
        $dumpvars(0, mxm_2x2_cocotb_top);
    end
`endif

endmodule
