`timescale 1ns/1ps
import lpu_pkg::*;

module mxm_mem_westbound_cocotb_top (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        mem_write_en,
    input  logic        mem_read_en,
    input  logic [8:0]  mem_addr,
    input  logic [31:0] mem_write_data,

    input logic         mem1_write_en, 
    input logic         mem1_read_en, 
    input logic [8:0]   mem1_addr,
    input logic [31:0]  mem1_write_data,

    input  logic [2:0]  westbound_sel,
    input  logic [2:0]  westbound_consumer_sel,
    input  logic [1:0]  mxm_ingress_mode,

    input  logic        mxm_clear,
    input  logic        mxm_start,

    output logic signed [31:0] mxm_out_00_dbg,
    output logic signed [31:0] mxm_out_01_dbg,
    output logic signed [31:0] mxm_out_02_dbg,
    output logic signed [31:0] mxm_out_03_dbg,
    output logic signed [31:0] mxm_out_10_dbg,
    output logic signed [31:0] mxm_out_11_dbg,
    output logic signed [31:0] mxm_out_12_dbg,
    output logic signed [31:0] mxm_out_13_dbg,
    output logic signed [31:0] mxm_out_20_dbg,
    output logic signed [31:0] mxm_out_21_dbg,
    output logic signed [31:0] mxm_out_22_dbg,
    output logic signed [31:0] mxm_out_23_dbg,
    output logic signed [31:0] mxm_out_30_dbg,
    output logic signed [31:0] mxm_out_31_dbg,
    output logic signed [31:0] mxm_out_32_dbg,
    output logic signed [31:0] mxm_out_33_dbg,
    output logic [31:0] westbound_payload_dbg,
    output logic        westbound_valid_dbg,
    output logic        mxm_west_en_dbg,
    output logic        act_loaded_dbg,
    output logic signed [7:0] act_buf0,
    output logic signed [7:0] act_buf1,
    output logic signed [7:0] act_buf2,
    output logic signed [7:0] act_buf3,

    output logic        wght_loaded_dbg, 
    output logic signed [7:0] wght_buf0,
    output logic signed [7:0] wght_buf1, 
    output logic signed [7:0] wght_buf2, 
    output logic signed [7:0] wght_buf3
);

    westbound_producer_e producer_sel_t;
    westbound_consumer_e consumer_sel_t;

    logic [31:0] mem0_stream_out;
    logic        mem0_valid;

    logic [31:0] mem1_stream_out;
    logic        mem1_valid;

    logic [31:0] westbound_payload;
    logic        westbound_valid;
    logic        mxm_west_en;

    logic signed [7:0]  mxm_act_in [3:0];
    logic               wght_load  [3:0];
    logic signed [7:0]  wght_val   [3:0];
    logic signed [31:0] mxm_out    [3:0][3:0];

    assign producer_sel_t = westbound_producer_e'(westbound_sel);
    assign consumer_sel_t = westbound_consumer_e'(westbound_consumer_sel);

    generate
        for (genvar i = 0; i < 4; i++) begin : g_tieoff
            assign mxm_act_in[i] = '0;
            assign wght_load[i]  = 1'b0;
            assign wght_val[i]   = '0;
        end
    endgenerate

    // mem_block read data is synchronous, so register valid alongside it.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
        begin 
            mem0_valid <= 1'b0;
            mem1_valid <= 1'b0;
        end 
        else
        begin 
            mem0_valid <= mem_read_en;
            mem1_valid <= mem1_read_en;
        end 
    end


    mem_block u_mem0 (
        .clk(clk),
        .rst_n(rst_n),
        .stream_in(mem_write_data),
        .stream_out(mem0_stream_out),
        .read_en(mem_read_en),
        .write_en(mem_write_en),
        .addr(mem_addr)
    );

    mem_block u_mem1 (
        .clk(clk), 
        .rst_n(rst_n),
        .stream_in(mem1_write_data),
        .stream_out(mem1_stream_out),
        .read_en(mem1_read_en),
        .write_en(mem1_write_en),
        .addr(mem1_addr)
    );

    westbound_bus #(
        .PAYLOAD_W(32)
    ) u_westbound_bus (
        .producer_sel(producer_sel_t),
        .sxm_payload('0),
        .sxm_valid(1'b0),
        .mem0_payload(mem0_stream_out),
        .mem0_valid(mem0_valid),
        .vxm_payload('0),
        .vxm_valid(1'b0),
        .mem1_payload(mem1_stream_out),
        .mem1_valid(mem1_valid),
        .westbound_payload(westbound_payload),
        .westbound_valid(westbound_valid)
    );

    westbound_consumer_decode u_westbound_consumer_decode (
        .consumer_sel(consumer_sel_t),
        .westbound_valid(westbound_valid),
        .mxm_west_en(mxm_west_en),
        .sxm_west_en(),
        .mem0_west_en(),
        .vxm_west_en()
    );

    mxm #(
        .mxm_size(4)
    ) u_mxm (
        .clk(clk),
        .rst(!rst_n),
        .mxm_clear(mxm_clear),
        .mxm_start(mxm_start),
        .westbound_payload(westbound_payload),
        .westbound_valid(westbound_valid),
        .mxm_west_en(mxm_west_en),
        .mxm_ingress_mode(mxm_ingress_mode),
        .mxm_act_in(mxm_act_in),
        .wght_load(wght_load),
        .wght_val(wght_val),
        .mxm_out(mxm_out)
    );

    assign mxm_out_00_dbg = mxm_out[0][0];
    assign mxm_out_01_dbg = mxm_out[0][1];
    assign mxm_out_02_dbg = mxm_out[0][2];
    assign mxm_out_03_dbg = mxm_out[0][3];
    assign mxm_out_10_dbg = mxm_out[1][0];
    assign mxm_out_11_dbg = mxm_out[1][1];
    assign mxm_out_12_dbg = mxm_out[1][2];
    assign mxm_out_13_dbg = mxm_out[1][3];
    assign mxm_out_20_dbg = mxm_out[2][0];
    assign mxm_out_21_dbg = mxm_out[2][1];
    assign mxm_out_22_dbg = mxm_out[2][2];
    assign mxm_out_23_dbg = mxm_out[2][3];
    assign mxm_out_30_dbg = mxm_out[3][0];
    assign mxm_out_31_dbg = mxm_out[3][1];
    assign mxm_out_32_dbg = mxm_out[3][2];
    assign mxm_out_33_dbg = mxm_out[3][3];

    assign westbound_payload_dbg = westbound_payload;
    assign westbound_valid_dbg   = westbound_valid;
    assign mxm_west_en_dbg       = mxm_west_en;

    assign act_loaded_dbg = u_mxm.mxm_act_ingress_loaded;
    assign act_buf0 = u_mxm.mxm_act_ingress_reg[0];
    assign act_buf1 = u_mxm.mxm_act_ingress_reg[1];
    assign act_buf2 = u_mxm.mxm_act_ingress_reg[2];
    assign act_buf3 = u_mxm.mxm_act_ingress_reg[3];

    assign wght_loaded_dbg = u_mxm.mxm_wght_ingress_loaded;
    assign wght_buf0 = u_mxm.mxm_wght_ingress_reg[0];
    assign wght_buf1 = u_mxm.mxm_wght_ingress_reg[1];
    assign wght_buf2 = u_mxm.mxm_wght_ingress_reg[2];
    assign wght_buf3 = u_mxm.mxm_wght_ingress_reg[3];

`ifdef WAVEFORM
    initial begin
        $dumpfile("mxm_mem_westbound_cocotb.vcd");
        $dumpvars(0, mxm_mem_westbound_cocotb_top);
    end
`endif

endmodule
