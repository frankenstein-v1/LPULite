`timescale 1ns/1ps

module mxm#(
    parameter int mxm_size = 4
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
    input logic signed [7:0] mxm_act_in [mxm_size-1 : 0],
    input logic wght_load [mxm_size-1 : 0],
    input logic signed [7:0] wght_val [mxm_size-1 : 0],
   

    //outputs
    output logic signed [31:0] mxm_out [mxm_size-1 : 0][mxm_size-1 : 0]
);

//wires that hold the product of each MAC units 
logic signed [19:0] product_wire [mxm_size-1:0][mxm_size-1:0];

//wire that holds the activation vector that came in from the wb bus
logic signed [7:0] mxm_act_ingress_reg [mxm_size-1:0];

//this act register contains valid data
logic mxm_act_ingress_loaded;

//wire that holds weight vector that came in from wb bus
logic signed [7:0] mxm_wght_ingress_reg [mxm_size-1:0];

//this wght register holds valid data
logic mxm_wght_ingress_loaded;

//act feed is what drives the data instead of the register wire, to give more flexibility so we can choose the source of the feed 
logic signed [7:0] mxm_act_feed [mxm_size-1:0];

//same with wght feed
logic signed [7:0] mxm_wght_feed [mxm_size-1:0];

//load command for wght vectors, its seperate so we can load the weight vectors 
logic [mxm_size-1:0] mxm_wght_load_feed;

//mxm should capture the wb bus this cycle
logic mxm_west_capture;

//indicates the oncoming bus payload should be treated as act
logic mxm_ingress_is_act;

logic mxm_ingress_is_wght;

//if west enable and westbound data is valid, capture the data
assign mxm_west_capture = mxm_west_en && westbound_valid;

//assign inputs as activations if activations is selected 
assign mxm_ingress_is_act = (mxm_ingress_mode == 2'b01);

//if the select bits is weight, load it into weight register
assign mxm_ingress_is_wght = (mxm_ingress_mode == 2'b10);

always_ff @(posedge clk or posedge rst) begin 
    //if reset or mxm_clear, we know the registers hjave nothing in them/ we reset them to 0
    if (rst || mxm_clear) begin 
        mxm_act_ingress_loaded <= 1'b0;
        mxm_wght_ingress_loaded <= 1'b0;


        //
        for (int idx = 0; idx < mxm_size; idx++) begin 
            mxm_act_ingress_reg[idx] <= '0;
            mxm_wght_ingress_reg[idx] <= '0;
        end 
    end 

    //if the mxm_West_capture 
    else if (mxm_west_capture) begin

        //if the select bits are for activation buffer  
        if (mxm_ingress_is_act) begin 
            for (int idx = 0; idx < mxm_size; idx++) begin
                mxm_act_ingress_reg[idx] <= westbound_payload[8*idx +: 8];//registers 0-3 in this case will hold the data in increments of 8 starting from 0
            end 
            //once loaded the register loaded should be 1
            mxm_act_ingress_loaded <= 1'b1;
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


always_comb begin 
    for (int idx = 0; idx < mxm_size; idx++) begin 
        // Use the current bus word immediately on a capture cycle so the ingress
        // path and the MAC weight-load pulse stay aligned.
        if (mxm_west_capture && mxm_ingress_is_act)
            mxm_act_feed[idx] = westbound_payload[8*idx +: 8];
        else if (mxm_act_ingress_loaded)//if register is loaded, the activations should be used from there
            mxm_act_feed[idx] = mxm_act_ingress_reg[idx];
        else
            mxm_act_feed[idx] = mxm_act_in[idx]; //otherwuise yse the external input

        if (mxm_west_capture && mxm_ingress_is_wght)
            mxm_wght_feed[idx] = westbound_payload[8*idx +: 8];
        else if (mxm_wght_ingress_loaded)
            mxm_wght_feed[idx] = mxm_wght_ingress_reg[idx];
        else
            mxm_wght_feed[idx] = wght_val[idx];
    end 
end 

always_comb begin 
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



genvar r, c;
generate
    for(r = 0; r < mxm_size; r++) begin: row
        for(c = 0; c<mxm_size; c++) begin : col
        mac u_mac(
            .clk(clk),
            .rst(rst),
            .en(mxm_start),
            .activation_in(mxm_act_feed[r]),
            .weight_load(mxm_wght_load_feed[c]),
            .weight_value(mxm_wght_feed[c]),
            .product(product_wire[r][c])
        );

        acc u_acc(
            .clk(clk),
            .rst(rst),
            .en(mxm_start),
            .clear(mxm_clear),
            .product_in(product_wire[r][c]),
            .sum_out(mxm_out[r][c])
        );


        end 
    end 


endgenerate

   
endmodule
