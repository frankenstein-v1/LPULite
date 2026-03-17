`timescale 1ns/1ps

module mxm#(
    parameter int mxm_size = 4
)(
    input logic clk, 
    input logic rst, 

    //input bits 
    input logic mxm_start,
    input logic [7:0] mxm_act_in [mxm_size-1 : 0],
    input logic wght_load [mxm_size-1 : 0],
    input logic [7:0] wght_val [mxm_size-1 : 0],

    //outputs
    output logic signed [19:0] mxm_out [mxm_size-1 : 0][mxm_size-1 : 0]
);

genvar r, c;
generate
    for(r = 0; r < mxm_size; r++) begin: row
        for(c = 0; c<mxm_size; c++) begin : col
        mac u_mac(
            .clk(clk),
            .rst(rst),
            .en(mxm_start),
            .activation_in(mxm_act_in[r]),
            .weight_load(wght_load[c]),
            .weight_value(wght_val[c]),
            .product(mxm_out[r][c])
        );

        end 
    end 


endgenerate



   
endmodule
