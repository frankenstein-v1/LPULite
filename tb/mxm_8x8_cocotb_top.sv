`timescale 1ns/1ps

module mxm_8x8_cocotb_top (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    mxm_clear,
    input  logic                    mxm_start,
    input  logic [63:0]             input_vec,
    input  logic signed [7:0]       input_scale,
    input  logic [63:0]             weight_vec,
    input  logic signed [7:0]       weight_scale,
    input  logic [7:0]              wght_load,
    output logic [(8*8*32)-1:0]     mxm_out_flat,
    output logic signed [7:0]       mxm_out_scale
);

    logic signed [7:0][7:0]       mxm_input_in;
    logic signed [7:0][7:0]       wght_val;
    logic signed [7:0][7:0][31:0] mxm_out;

    genvar i, r, c;
    generate
        for (i = 0; i < 8; i++) begin : unpack_lanes
            assign mxm_input_in[i] = input_vec[8*i +: 8];
            assign wght_val[i]     = weight_vec[8*i +: 8];
        end

        for (r = 0; r < 8; r++) begin : pack_rows
            for (c = 0; c < 8; c++) begin : pack_cols
                assign mxm_out_flat[32*(8*r+c) +: 32] = mxm_out[r][c];
            end
        end
    endgenerate

    mxm #(
        .mxm_size(8)
    ) dut (
        .clk(clk),
        .rst(rst),
        .mxm_clear(mxm_clear),
        .mxm_start(mxm_start),
        .westbound_payload('0),
        .westbound_valid(1'b0),
        .mxm_west_en(1'b0),
        .mxm_ingress_mode(2'b00),
        .mxm_use_fp(1'b0),
        .mxm_input_is_signed(1'b1),
        .mxm_wght_is_signed(1'b1),
        .mxm_input_in(mxm_input_in),
        .mxm_input_scale_i(input_scale),
        .wght_load(wght_load),
        .wght_val(wght_val),
        .mxm_wght_scale_i(weight_scale),
        .mxm_out(mxm_out),
        .mxm_out_scale_o(mxm_out_scale)
    );

endmodule
