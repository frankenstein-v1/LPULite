`timescale 1ns/1ps

module microgpt_lpu_top (
    input  logic        clk,
    input  logic        rst,
    input  logic        clear,
    input  logic        start,
    input  logic [63:0] payload,
    input  logic        payload_valid,
    input  logic        mxm_enable,
    input  logic [1:0]  ingress_mode,
    input  logic signed [7:0] input_scale,
    input  logic signed [7:0] weight_scale,
    output logic signed [31:0] result0,
    output logic signed [31:0] result1,
    output logic signed [31:0] result2,
    output logic signed [31:0] result3,
    output logic signed [31:0] result4,
    output logic signed [31:0] result5,
    output logic signed [31:0] result6,
    output logic signed [31:0] result7,
    output logic signed [7:0]  result_scale
);

    logic signed [7:0][7:0] unused_inputs;
    logic signed [7:0][7:0] unused_weights;
    logic signed [7:0][7:0][31:0] results;

    mxm u_mxm (
        .clk,
        .rst,
        .mxm_clear(clear),
        .mxm_start(start),
        .westbound_payload(payload),
        .westbound_valid(payload_valid),
        .mxm_west_en(mxm_enable),
        .mxm_ingress_mode(ingress_mode),
        .mxm_input_is_signed(1'b1),
        .mxm_wght_is_signed(1'b1),
        .mxm_input_in(unused_inputs),
        .mxm_input_scale_i(input_scale),
        .wght_load(8'b0),
        .wght_val(unused_weights),
        .mxm_wght_scale_i(weight_scale),
        .mxm_out(results),
        .mxm_out_scale_o(result_scale)
    );

    assign result0 = results[0][0];
    assign result1 = results[0][1];
    assign result2 = results[0][2];
    assign result3 = results[0][3];
    assign result4 = results[0][4];
    assign result5 = results[0][5];
    assign result6 = results[0][6];
    assign result7 = results[0][7];

endmodule
