`timescale 1ns/1ps

module mxm#(
    parameter int mxm_size = 8
)(
    input logic clk, 
    input logic rst, 
    input logic mxm_clear,
    input logic mxm_start,

    //input bits 
    input logic [8  * mxm_size - 1 : 0] westbound_payload,
    input logic westbound_valid, 
    input logic mxm_west_en, 
    input logic [1:0] mxm_ingress_mode, 
    input logic mxm_input_is_signed,
    input logic mxm_wght_is_signed,
    input logic signed [mxm_size-1 : 0][7:0] mxm_input_in,
    input logic signed [7:0] mxm_input_scale_i,
    input logic [mxm_size-1 : 0] wght_load,
    input logic signed [mxm_size-1 : 0][7:0] wght_val,
    input logic signed [7:0] mxm_wght_scale_i,
   

    //outputs
    output logic signed [mxm_size-1 : 0][mxm_size-1 : 0][31:0] mxm_out,
    output logic signed [7:0] mxm_out_scale_o
);

localparam int MAC_OPERAND_W = 9;
localparam int MAC_PRODUCT_W = 2 * MAC_OPERAND_W;
localparam int MAC_ACC_W     = 32;
localparam int MAC_SCALE_W   = 8;

//wires that hold the product of each MAC units
logic signed [MAC_PRODUCT_W-1:0] product_wire [mxm_size-1:0][mxm_size-1:0];
logic signed [MAC_ACC_W-1:0]     mac_accum_wire [mxm_size-1:0][mxm_size-1:0];
logic signed [MAC_SCALE_W-1:0]   mac_acc_scale_wire [mxm_size-1:0][mxm_size-1:0];

//wire that holds the input vector that came in from the wb bus
logic signed [7:0] mxm_input_ingress_reg [mxm_size-1:0];
logic signed [MAC_SCALE_W-1:0] mxm_input_scale_reg;

//this input register contains valid data
logic mxm_input_ingress_loaded;

//wire that holds weight vector that came in from wb bus
logic signed [7:0] mxm_wght_ingress_reg [mxm_size-1:0];

//this wght register holds valid data
logic mxm_wght_ingress_loaded;

//input feed is what drives the data instead of the register wire, to give more flexibility so we can choose the source of the feed 
logic signed [7:0] mxm_input_feed [mxm_size-1:0];

//same with wght feed
logic signed [7:0] mxm_wght_feed [mxm_size-1:0];

//registered weight values used by the MAC array
logic signed [MAC_OPERAND_W-1:0] mxm_wght_reg [mxm_size-1:0];
logic signed [MAC_SCALE_W-1:0]   mxm_wght_scale_reg;

//sign/zero-extended operands driven into the MAC array
logic signed [MAC_OPERAND_W-1:0] mxm_input_mac_feed [mxm_size-1:0];
logic signed [MAC_OPERAND_W-1:0] mxm_wght_mac_feed [mxm_size-1:0];

//load command for wght vectors, its seperate so we can load the weight vectors 
logic [mxm_size-1:0] mxm_wght_load_feed;

//old MXM timing had one registered product stage before accumulation
logic [mxm_size-1:0] mxm_mac_en_feed;

//mxm should capture the wb bus this cycle
logic mxm_west_capture;

//indicates the oncoming bus payload should be treated as input
logic mxm_ingress_is_input;

logic mxm_ingress_is_wght;

function automatic logic signed [MAC_OPERAND_W-1:0] extend_operand(
    input logic [7:0] operand,
    input logic       operand_is_signed
);
    begin
        extend_operand = operand_is_signed
            ? $signed({operand[7], operand})
            : $signed({1'b0, operand});
    end
endfunction

//if west enable and westbound data is valid, capture the data
assign mxm_west_capture = mxm_west_en && westbound_valid;

//assign inputs if input mode is selected 
assign mxm_ingress_is_input = (mxm_ingress_mode == 2'b01);

//if the select bits is weight, load it into weight register
assign mxm_ingress_is_wght = (mxm_ingress_mode == 2'b10);

always_ff @(posedge clk or posedge rst) begin 
    // rst is asynchronous; mxm_clear is a synchronous command.
    if (rst) begin 
        mxm_input_ingress_loaded <= 1'b0;
        mxm_wght_ingress_loaded <= 1'b0;

        for (int idx = 0; idx < mxm_size; idx++) begin 
            mxm_input_ingress_reg[idx] <= '0;
            mxm_wght_ingress_reg[idx] <= '0;
        end
        mxm_input_scale_reg <= '0;
    end else if (mxm_clear) begin
        mxm_input_ingress_loaded <= 1'b0;
        mxm_wght_ingress_loaded <= 1'b0;


        //
        for (int idx = 0; idx < mxm_size; idx++) begin 
            mxm_input_ingress_reg[idx] <= '0;
            mxm_wght_ingress_reg[idx] <= '0;
        end 
        mxm_input_scale_reg <= '0;
    end 

    //if the mxm_West_capture 
    else if (mxm_west_capture) begin

        //if the select bits are for input buffer  
        if (mxm_ingress_is_input) begin 
            for (int idx = 0; idx < mxm_size; idx++) begin
                mxm_input_ingress_reg[idx] <= westbound_payload[8*idx +: 8];//registers 0-3 in this case will hold the data in increments of 8 starting from 0
            end 
            //once loaded the register loaded should be 1
            mxm_input_ingress_loaded <= 1'b1;
            mxm_input_scale_reg <= mxm_input_scale_i;
        end 

        else if (mxm_ingress_is_wght) begin 
            for (int idx = 0; idx < mxm_size; idx++) begin 
                //howevver in this case of weight register being used, same thing the data coming in is 32 bits because 4 MACs and 8 bit, so each reg is 8 bits
                mxm_wght_ingress_reg[idx] <= westbound_payload[8*idx +: 8];
            end 
            //once loaded the register loaded should be 1
            mxm_wght_ingress_loaded <= 1'b1;
        end 
    end 
end 

always @* begin 
    for (int idx = 0; idx < mxm_size; idx++) begin 
        // Use the current bus word immediately on a capture cycle so the ingress
        // path and the MAC weight-load pulse stay aligned.
        if (mxm_west_capture && mxm_ingress_is_input)
            mxm_input_feed[idx] = westbound_payload[8*idx +: 8];
        else if (mxm_input_ingress_loaded)//if register is loaded, the inputs should be used from there
            mxm_input_feed[idx] = mxm_input_ingress_reg[idx];
        else
            mxm_input_feed[idx] = mxm_input_in[idx]; //otherwise use the external input

        if (mxm_west_capture && mxm_ingress_is_wght)
            mxm_wght_feed[idx] = westbound_payload[8*idx +: 8];
        else if (mxm_wght_ingress_loaded)
            mxm_wght_feed[idx] = mxm_wght_ingress_reg[idx];
        else
            mxm_wght_feed[idx] = wght_val[idx];
    end 
end 

always @* begin
    for (int idx = 0; idx < mxm_size; idx++) begin
        mxm_input_mac_feed[idx] = extend_operand(mxm_input_feed[idx], mxm_input_is_signed);
        mxm_wght_mac_feed[idx] = mxm_wght_load_feed[idx]
            ? extend_operand(mxm_wght_feed[idx], mxm_wght_is_signed)
            : mxm_wght_reg[idx];
    end
end

always @* begin 
    mxm_wght_load_feed = '0;

    if (mxm_west_capture && mxm_ingress_is_wght) begin 
        for (int idx = 0; idx < mxm_size; idx++) begin 
            mxm_wght_load_feed[idx] = 1'b1;
        end 
    end else begin
        for (int idx = 0; idx < mxm_size; idx++) begin
            mxm_wght_load_feed[idx] = wght_load[idx];
        end
    end 
end 

always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        for (int idx = 0; idx < mxm_size; idx++) begin
            mxm_wght_reg[idx] <= '0;
        end
        mxm_wght_scale_reg <= '0;
        mxm_mac_en_feed <= '0;
    end else if (mxm_clear) begin
        for (int idx = 0; idx < mxm_size; idx++) begin
            mxm_wght_reg[idx] <= '0;
        end
        mxm_wght_scale_reg <= '0;
        mxm_mac_en_feed <= '0;
    end else begin
        for (int idx = 0; idx < mxm_size; idx++) begin
            mxm_mac_en_feed[idx] <= mxm_start && !mxm_wght_load_feed[idx];
        end

        for (int idx = 0; idx < mxm_size; idx++) begin
            if (mxm_wght_load_feed[idx]) begin
                mxm_wght_reg[idx] <= extend_operand(mxm_wght_feed[idx], mxm_wght_is_signed);
            end
        end

        if (|mxm_wght_load_feed) begin
            mxm_wght_scale_reg <= mxm_wght_scale_i;
        end
    end
end



genvar r, c;
generate
    for(r = 0; r < mxm_size; r++) begin: row
        for(c = 0; c<mxm_size; c++) begin : col
            mac #(
                .INPUT_W(MAC_OPERAND_W),
                .WEIGHT_W(MAC_OPERAND_W),
                .PRODUCT_W(MAC_PRODUCT_W),
                .ACC_W(MAC_ACC_W),
                .SCALE_W(MAC_SCALE_W)
            ) u_mac (
                .clk(clk),
                .rst(rst),
                .clear(mxm_clear),
                .en(mxm_mac_en_feed[c]),
                .input_i(mxm_input_mac_feed[r]),
                .weight_i(mxm_wght_mac_feed[c]),
                .input_scale_i(mxm_input_ingress_loaded ? mxm_input_scale_reg : mxm_input_scale_i),
                .weight_scale_i(mxm_wght_scale_reg),
                .acc_o(mac_accum_wire[r][c]),
                .acc_scale_o(mac_acc_scale_wire[r][c]),
                .product_o(product_wire[r][c])
            );

            assign mxm_out[r][c] = mac_accum_wire[r][c];
        end 
    end 


endgenerate

assign mxm_out_scale_o = mac_acc_scale_wire[0][0];
   
endmodule
