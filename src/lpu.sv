`timescale 1ns/1ps
import lpu_pkg::*;

module lpu (
    input logic clk,
    input logic rst_n
);

//icu mem outputs
logic mem0_read_en;
logic mem0_write_en;
logic [8:0] mem0_addr;
logic mem1_read_en;
logic mem1_write_en;
logic [8:0] mem1_addr;

//icu sxm outputs
logic [11:0] sxm_opcode_input;
logic [11:0] sxm_opcode_weight;

//icxu vxm outs
logic [1:0] vxm_math_op;
logic vxm_accum_en;
logic vxm_flush;

//icu busses
logic [2:0] westbound_sel;
logic [2:0] eastbound_sel;
logic [2:0] westbound_consumer_sel;
logic  [2:0] eastbound_consumer_sel;

//icu mxm outs
logic [1:0] mxm_ingress_mode;
logic mxm_start;
logic mxm_clear;
logic [1:0] mxm_e_row_sel;
logic [1:0] mxm_e_col_sel;
logic mxm_e_valid_in;

// typed bus select views
westbound_consumer_e westbound_consumer_sel_t;
eastbound_consumer_e eastbound_consumer_sel_t;

westbound_producer_e westbound_sel_t;
eastbound_producer_e eastbound_sel_t;

// shared bus signals (to be driven by bus fabric later)
superlane_t westbound_payload;
logic       westbound_valid;
superlane_t eastbound_payload;
logic       eastbound_valid;

// mem0 datapath
superlane_t mem0_stream_in;
superlane_t mem0_stream_out;
logic       mem0_write_from_west;
logic       mem0_write_from_east;
logic       mem0_write_en_eff;

logic mem0_valid;
logic mem1_valid;

logic mxm_west_en;
logic sxm_west_en;
logic mem0_west_en;
logic vxm_west_en;

// mxm datapath
logic signed [3:0][7:0]  mxm_input_in;
logic [3:0]              wght_load;
logic signed [3:0][7:0]  wght_val;
logic signed [3:0][3:0][31:0] mxm_out;

//icu instance
icu u_icu(
    .clk(clk),
    .rst_n(rst_n),
    .mem0_read_en(mem0_read_en),
    .mem0_write_en(mem0_write_en),
    .mem0_addr(mem0_addr),
    .mem1_read_en(mem1_read_en),
    .mem1_write_en(mem1_write_en),
    .mem1_addr(mem1_addr),
    .sxm_opcode_input(sxm_opcode_input),
    .sxm_opcode_weight(sxm_opcode_weight),
    .vxm_math_op(vxm_math_op),
    .vxm_accum_en(vxm_accum_en),
    .vxm_flush(vxm_flush),
    .westbound_sel(westbound_sel),
    .eastbound_sel(eastbound_sel),
    .westbound_consumer_sel(westbound_consumer_sel),
    .eastbound_consumer_sel(eastbound_consumer_sel),
    .mxm_ingress_mode(mxm_ingress_mode),
    .mxm_start(mxm_start),
    .mxm_clear(mxm_clear),
    .mxm_e_row_sel(mxm_e_row_sel),
    .mxm_e_col_sel(mxm_e_col_sel),
    .mxm_e_valid_in(mxm_e_valid_in)
);

//mux logic for mem0 input

assign westbound_consumer_sel_t = westbound_consumer_e'(westbound_consumer_sel);
assign eastbound_consumer_sel_t = eastbound_consumer_e'(eastbound_consumer_sel);

assign mem0_write_from_west =
    mem0_write_en && (westbound_consumer_sel_t == WC_MEM0) && westbound_valid;

assign mem0_write_from_east =
    mem0_write_en && (eastbound_consumer_sel_t == EC_MEM0) && eastbound_valid;

assign mem0_write_en_eff = mem0_write_from_west || mem0_write_from_east;

always_comb begin
    mem0_stream_in = '0;

    if (mem0_write_from_west)
        mem0_stream_in = westbound_payload;
    else if (mem0_write_from_east)
        mem0_stream_in = eastbound_payload;
end

mem u_mem0(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in(mem0_stream_in),
    .stream_out(mem0_stream_out),
    .read_en(mem0_read_en),
    .write_en(mem0_write_en_eff),
    .addr(mem0_addr)
);

//since mem1 is on the far left westbound cannot write into it, so only eastbound can 
superlane_t mem1_stream_out;

logic mem1_write_en_eff;

assign mem1_write_en_eff = mem1_write_en && (eastbound_consumer_sel_t == EC_MEM1) && eastbound_valid;

assign westbound_sel_t = westbound_producer_e'(westbound_sel);

assign eastbound_sel_t = eastbound_producer_e'(eastbound_sel);

mem u_mem1(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in(eastbound_payload),
    .stream_out(mem1_stream_out),
    .read_en(mem1_read_en),
    .write_en(mem1_write_en_eff),
    .addr(mem1_addr)
);

// memory read data is synchronous, so register valid alongside the read enable
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        mem0_valid <= 1'b0;
        mem1_valid <= 1'b0;
    end else begin
        mem0_valid <= mem0_read_en;
        mem1_valid <= mem1_read_en;
    end
end

westbound_bus u_westbound_bus(
    .producer_sel(westbound_sel_t),
    .mem0_payload(mem0_stream_out),
    .mem1_payload(mem1_stream_out),
    .mem0_valid(mem0_valid),
    .mem1_valid(mem1_valid),
    .sxm_payload('0),
    .sxm_valid(1'b0), 
    .vxm_payload('0),
    .vxm_valid(1'b0),
    .westbound_payload(westbound_payload),
    .westbound_valid(westbound_valid)
);

westbound_consumer_decode u_westbound_consumer_decode(
    .consumer_sel(westbound_consumer_sel_t),
    .westbound_valid(westbound_valid),
    .mxm_west_en(mxm_west_en),
    .sxm_west_en(sxm_west_en),
    .mem0_west_en(mem0_west_en),
    .vxm_west_en(vxm_west_en)
);

logic [31:0] mxm_payload_e;
logic mxm_valid_e;

mxm_eastbound_adapter #(
    .MXM_SIZE(4),
    .PAYLOAD_W(32)
) u_mxm_to_eastbound (
    .mxm_out(mxm_out),
    .mxm_row_sel(mxm_e_row_sel),
    .mxm_col_sel(mxm_e_col_sel),
    .mxm_valid_in(mxm_e_valid_in),
    .mxm_payload(mxm_payload_e),
    .mxm_valid(mxm_valid_e)
);

eastbound_bus #(
    .PAYLOAD_E(32)
) u_eastbound_bus(
    .producer_sel(eastbound_sel_t),
    .mxm_payload_e(mxm_payload_e),
    .mxm_valid_e(mxm_valid_e),
    .vxm_payload_e('0),
    .vxm_valid_e(1'b0),
    .sxm_payload_e('0),
    .sxm_valid_e(1'b0),
    .mem0_payload_e(mem0_stream_out),
    .mem0_valid_e(mem0_valid),
    .eastbound_payload(eastbound_payload),
    .eastbound_valid(eastbound_valid)
);

logic sxm_east_en;
logic mem0_east_en;
logic vxm_east_en;
logic mem1_east_en;

eastbound_consumer_decode u_eastbound_consumer_decode(
    .consumer_sel(eastbound_consumer_sel_t),
    .eastbound_valid(eastbound_valid),
    .sxm_east_en(sxm_east_en),
    .mem0_east_en(mem0_east_en),
    .vxm_east_en(vxm_east_en),
    .mem1_east_en(mem1_east_en)
);

logic [31:0] sxm_e_payload_reg;
logic [31:0] sxm_w_payload_reg;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sxm_e_payload_reg <= '0;
        sxm_w_payload_reg <= '0;
    end else begin
        if (sxm_east_en && eastbound_valid)
            sxm_e_payload_reg <= eastbound_payload;
        if (sxm_west_en && westbound_valid)
            sxm_w_payload_reg <= westbound_payload;
    end
end

superlane_t sxm_stream_out_to_mxm_left;
superlane_t sxm_stream_out_to_mxm_top;

sxm u_sxm(
    .clk(clk),
    .rst_n(rst_n),
    .opcode_input(sxm_opcode_input),
    .opcode_weight(sxm_opcode_weight),
    .eastbound_in(sxm_e_payload_reg),
    .westbound_in(sxm_w_payload_reg),
    .eastbound_out(sxm_stream_out_to_mxm_left),
    .westbound_out(sxm_stream_out_to_mxm_top)
);

generate
    for (genvar i = 0; i < 4; i++) begin : g_mxm_feed
        assign mxm_input_in[i] = sxm_stream_out_to_mxm_left[i*8 +: 8];
        assign wght_val[i]     = sxm_stream_out_to_mxm_top[i*8 +: 8];
        assign wght_load[i]    = 1'b0;
    end
endgenerate

mxm u_mxm(
    .clk(clk),
    .rst(~rst_n),
    .mxm_clear(mxm_clear),
    .mxm_start(mxm_start),
    .westbound_payload(westbound_payload),
    .westbound_valid(westbound_valid),
    .mxm_west_en(mxm_west_en),
    .mxm_ingress_mode(mxm_ingress_mode),
    .mxm_input_in(mxm_input_in),
    .wght_load(wght_load),
    .wght_val(wght_val),
    .mxm_out(mxm_out)
);





endmodule
