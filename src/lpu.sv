`timescale 1ns/1ps
`include "lpu_pkg.sv"

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
logic [3:0] vxm_ctrl;
logic vxm_data_sel;

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
logic mxm_input_is_signed;
logic mxm_wght_is_signed;

// Encoded bus select views. Keep these as plain vectors for Icarus compatibility.
logic [2:0] westbound_consumer_sel_t;
logic [2:0] eastbound_consumer_sel_t;

logic [2:0] westbound_sel_t;
logic [2:0] eastbound_sel_t;

// shared bus signals (to be driven by bus fabric later)
superlane_t westbound_payload;
logic       westbound_valid;
mxm_row_t   eastbound_payload;
logic       eastbound_valid;
superlane_t eastbound_payload_lane0;

// mem0 datapath
mxm_row_t   mem0_stream_in;
mxm_row_t   mem0_stream_out;
logic       mem0_write_from_west;
logic       mem0_write_from_east;
logic       mem0_write_en_eff;

logic mem0_valid;
logic mem1_valid;

logic mxm_west_en;
logic sxm_west_en;
logic mem0_west_en;
logic vxm_west_en;

superlane_t sxm_stream_out_to_mxm_left;
superlane_t sxm_stream_out_to_mxm_top;

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
    .vxm_ctrl(vxm_ctrl),
    .vxm_data_sel(vxm_data_sel),
    .westbound_sel(westbound_sel),
    .eastbound_sel(eastbound_sel),
    .westbound_consumer_sel(westbound_consumer_sel),
    .eastbound_consumer_sel(eastbound_consumer_sel),
    .mxm_ingress_mode(mxm_ingress_mode),
    .mxm_start(mxm_start),
    .mxm_clear(mxm_clear),
    .mxm_e_row_sel(mxm_e_row_sel),
    .mxm_e_col_sel(mxm_e_col_sel),
    .mxm_e_valid_in(mxm_e_valid_in),
    .mxm_input_is_signed(mxm_input_is_signed),
    .mxm_wght_is_signed(mxm_wght_is_signed)
);

//mux logic for mem0 input

assign westbound_consumer_sel_t = westbound_consumer_sel;
assign eastbound_consumer_sel_t = eastbound_consumer_sel;

assign mem0_write_from_west =
    mem0_write_en && (westbound_consumer_sel_t == WC_MEM0) && westbound_valid;

assign mem0_write_from_east =
    mem0_write_en && (eastbound_consumer_sel_t == EC_MEM0) && eastbound_valid;

assign mem0_write_en_eff = mem0_write_from_west || mem0_write_from_east;

always_comb begin
    mem0_stream_in = '0;

    if (mem0_write_from_west)
        mem0_stream_in[31:0] = westbound_payload;
    else if (mem0_write_from_east)
        mem0_stream_in = eastbound_payload;
end

mem #(
    .DATA_W($bits(mxm_row_t))
) u_mem0(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in(mem0_stream_in),
    .stream_out(mem0_stream_out),
    .read_en(mem0_read_en),
    .write_en(mem0_write_en_eff),
    .addr(mem0_addr)
);

//since mem1 is on the far left westbound cannot write into it, so only eastbound can 
mxm_row_t mem1_stream_out;

logic mem1_write_en_eff;

assign mem1_write_en_eff = mem1_write_en && (eastbound_consumer_sel_t == EC_MEM1) && eastbound_valid;

assign westbound_sel_t = westbound_sel;

assign eastbound_sel_t = eastbound_sel;

mem #(
    .DATA_W($bits(mxm_row_t))
) u_mem1(
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

logic [31:0] vxm_stream_out_live;
logic [31:0] vxm_stream_out_scale_live;
logic        vxm_out_valid_live;
logic [31:0] vxm_stream_out_buf;
logic [31:0] vxm_stream_out_scale_buf;
logic        vxm_stream_out_buf_valid_w;
logic        vxm_stream_out_buf_valid_e;
logic        vxm_out_ready;
logic        vxm_result_wr_en;
logic        vxm_result_rd_en;
logic        vxm_result_full;
logic        vxm_result_empty;
logic        vxm_result_take_west;
logic        vxm_result_take_east;
logic        sxm_emit_valid;
logic        sxm_load_from_west;

westbound_bus u_westbound_bus(
    .producer_sel(westbound_sel_t),
    .mem0_payload(mem0_stream_out[31:0]),
    .mem1_payload(mem1_stream_out[31:0]),
    .mem0_valid(mem0_valid),
    .mem1_valid(mem1_valid),
    // SXM emits transpose rows onto westbound for downstream scheduling.
    .sxm_payload(sxm_stream_out_to_mxm_top),
    .sxm_valid(sxm_emit_valid),
    .vxm_payload(vxm_stream_out_buf),
    .vxm_valid(vxm_stream_out_buf_valid_w),
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

mxm_row_t mxm_payload_e;
logic mxm_valid_e;
mxm_row_t vxm_payload_e_bus;
mxm_row_t sxm_payload_e_bus;
mxm_row_t mem0_payload_e_bus;

assign eastbound_payload_lane0 = eastbound_payload[31:0];

always_comb begin
    vxm_payload_e_bus = '0;
    sxm_payload_e_bus = '0;
    mem0_payload_e_bus = '0;

    vxm_payload_e_bus[31:0] = vxm_stream_out_buf;
    vxm_payload_e_bus[63:32] = vxm_stream_out_scale_buf;
    sxm_payload_e_bus[31:0] = sxm_stream_out_to_mxm_left;
    mem0_payload_e_bus = mem0_stream_out;
end

mxm_eastbound_adapter #(
    .MXM_SIZE(MXM_SIZE),
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
    .PAYLOAD_E($bits(mxm_row_t))
) u_eastbound_bus(
    .producer_sel(eastbound_sel_t),
    .mxm_payload_e(mxm_payload_e),
    .mxm_valid_e(mxm_valid_e),
    .vxm_payload_e(vxm_payload_e_bus),
    .vxm_valid_e(vxm_stream_out_buf_valid_e),
    .sxm_payload_e(sxm_payload_e_bus),
    .sxm_valid_e(sxm_emit_valid),
    .mem0_payload_e(mem0_payload_e_bus),
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
            sxm_e_payload_reg <= eastbound_payload_lane0;
        if (sxm_west_en && westbound_valid)
            sxm_w_payload_reg <= westbound_payload;
    end
end

assign sxm_load_from_west = sxm_west_en && westbound_valid;

sxm u_sxm(
    .clk(clk),
    .rst_n(rst_n),
    .opcode_input(sxm_opcode_input),
    .opcode_weight(sxm_opcode_weight),
    .load_from_west(sxm_load_from_west),
    .eastbound_in(sxm_e_payload_reg),
    .westbound_in(sxm_w_payload_reg),
    .eastbound_out(sxm_stream_out_to_mxm_left),
    .westbound_out(sxm_stream_out_to_mxm_top),
    .emit_valid(sxm_emit_valid)
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
    .mxm_use_fp(1'b0),
    .mxm_input_is_signed(mxm_input_is_signed),
    .mxm_wght_is_signed(mxm_wght_is_signed),
    .mxm_input_in(mxm_input_in),
    .wght_load(wght_load),
    .wght_val(wght_val),
    .mxm_out(mxm_out)
);




mxm_row_t vxm_stream_in_data;
mxm_row_t vxm_stream_in_bias;
logic     vxm_in_valid;
logic     vxm_in_ready;
logic     vxm_fifo_wr_en;
logic     vxm_fifo_rd_en;
logic     vxm_fifo_full;
logic     vxm_fifo_empty;
mxm_row_t vxm_fifo_data_out;
logic     vxm_input_overflow;

// For now VXM consumes full-width rows from the eastbound bus only.
// ICU still controls whether that traffic appears on eastbound via the bus selectors.
assign vxm_fifo_wr_en = vxm_east_en && eastbound_valid;
assign vxm_fifo_rd_en = !vxm_fifo_empty && vxm_in_ready;
assign vxm_stream_in_data = vxm_fifo_data_out;
assign vxm_stream_in_bias = '0;
assign vxm_in_valid = !vxm_fifo_empty;

assign vxm_result_take_west = !vxm_result_empty && (westbound_sel_t == WB_VXM);
assign vxm_result_take_east = !vxm_result_empty &&
                              (westbound_sel_t != WB_VXM) &&
                              (eastbound_sel_t == EB_VXM);
assign vxm_result_rd_en = vxm_result_take_west || vxm_result_take_east;
assign vxm_stream_out_buf_valid_w = !vxm_result_empty;
assign vxm_stream_out_buf_valid_e = !vxm_result_empty && (westbound_sel_t != WB_VXM);
assign vxm_out_ready = !vxm_result_full;
assign vxm_result_wr_en = vxm_out_valid_live && vxm_out_ready;

row_fifo #(
    .DATA_W($bits(mxm_row_t)),
    .DEPTH(4)
) u_vxm_input_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(vxm_fifo_wr_en),
    .rd_en(vxm_fifo_rd_en),
    .data_in(eastbound_payload),
    .data_out(vxm_fifo_data_out),
    .full(vxm_fifo_full),
    .empty(vxm_fifo_empty)
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        vxm_input_overflow <= 1'b0;
    end else if (vxm_fifo_wr_en && vxm_fifo_full) begin
        vxm_input_overflow <= 1'b1;
    end
end

row_fifo #(
    .DATA_W(32),
    .DEPTH(4)
) u_vxm_output_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(vxm_result_wr_en),
    .rd_en(vxm_result_rd_en),
    .data_in(vxm_stream_out_live),
    .data_out(vxm_stream_out_buf),
    .full(vxm_result_full),
    .empty(vxm_result_empty)
);

row_fifo #(
    .DATA_W(32),
    .DEPTH(4)
) u_vxm_output_scale_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(vxm_result_wr_en),
    .rd_en(vxm_result_rd_en),
    .data_in(vxm_stream_out_scale_live),
    .data_out(vxm_stream_out_scale_buf),
    .full(),
    .empty()
);

vxm #(
    .LANES(4),
    .LANE_W(32),
    .ALU_W(32)
) u_vxm(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in_data(vxm_stream_in_data),
    .stream_in_bias(vxm_stream_in_bias),
    .in_valid(vxm_in_valid),
    .in_ready(vxm_in_ready),
    .vxm_ctrl(vxm_ctrl),
    .stream_out(vxm_stream_out_live),
    .stream_out_scale(vxm_stream_out_scale_live),
    .out_valid(vxm_out_valid_live),
    .out_ready(vxm_out_ready)
);

endmodule
