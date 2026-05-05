`timescale 1ns/1ps
import lpu_pkg::*;

module lpu_cocotb_top (
    input  logic        clk,
    input  logic        rst_n,

    output logic [31:0] pc_dbg,

    output logic        mem0_read_en_dbg,
    output logic        mem0_write_en_dbg,
    output logic [8:0]  mem0_addr_dbg,
    output logic        mem1_read_en_dbg,
    output logic        mem1_write_en_dbg,
    output logic [8:0]  mem1_addr_dbg,

    output logic [2:0]  westbound_sel_dbg,
    output logic [2:0]  westbound_consumer_sel_dbg,
    output logic [1:0]  mxm_ingress_mode_dbg,
    output logic        mxm_start_dbg,
    output logic        mxm_clear_dbg,

    output logic        mem0_valid_dbg,
    output logic        mem1_valid_dbg,
    output logic [31:0] westbound_payload_dbg,
    output logic        westbound_valid_dbg,
    output logic        mxm_west_en_dbg,

    output logic        input_loaded_dbg,
    output logic signed [7:0] input_buf0,
    output logic signed [7:0] input_buf1,
    output logic signed [7:0] input_buf2,
    output logic signed [7:0] input_buf3,

    output logic        wght_loaded_dbg,
    output logic signed [7:0] wght_buf0,
    output logic signed [7:0] wght_buf1,
    output logic signed [7:0] wght_buf2,
    output logic signed [7:0] wght_buf3,

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
    output logic signed [31:0] mxm_out_33_dbg
);

    lpu u_lpu (
        .clk(clk),
        .rst_n(rst_n)
    );

    assign pc_dbg                    = u_lpu.u_icu.pc;

    assign mem0_read_en_dbg          = u_lpu.mem0_read_en;
    assign mem0_write_en_dbg         = u_lpu.mem0_write_en;
    assign mem0_addr_dbg             = u_lpu.mem0_addr;
    assign mem1_read_en_dbg          = u_lpu.mem1_read_en;
    assign mem1_write_en_dbg         = u_lpu.mem1_write_en;
    assign mem1_addr_dbg             = u_lpu.mem1_addr;

    assign westbound_sel_dbg         = u_lpu.westbound_sel;
    assign westbound_consumer_sel_dbg = u_lpu.westbound_consumer_sel;
    assign mxm_ingress_mode_dbg      = u_lpu.mxm_ingress_mode;
    assign mxm_start_dbg             = u_lpu.mxm_start;
    assign mxm_clear_dbg             = u_lpu.mxm_clear;

    assign mem0_valid_dbg            = u_lpu.mem0_valid;
    assign mem1_valid_dbg            = u_lpu.mem1_valid;
    assign westbound_payload_dbg     = u_lpu.westbound_payload;
    assign westbound_valid_dbg       = u_lpu.westbound_valid;
    assign mxm_west_en_dbg           = u_lpu.mxm_west_en;

    assign input_loaded_dbg          = u_lpu.u_mxm.mxm_input_ingress_loaded;
    assign input_buf0                = u_lpu.u_mxm.mxm_input_ingress_reg[0];
    assign input_buf1                = u_lpu.u_mxm.mxm_input_ingress_reg[1];
    assign input_buf2                = u_lpu.u_mxm.mxm_input_ingress_reg[2];
    assign input_buf3                = u_lpu.u_mxm.mxm_input_ingress_reg[3];

    assign wght_loaded_dbg           = u_lpu.u_mxm.mxm_wght_ingress_loaded;
    assign wght_buf0                 = u_lpu.u_mxm.mxm_wght_ingress_reg[0];
    assign wght_buf1                 = u_lpu.u_mxm.mxm_wght_ingress_reg[1];
    assign wght_buf2                 = u_lpu.u_mxm.mxm_wght_ingress_reg[2];
    assign wght_buf3                 = u_lpu.u_mxm.mxm_wght_ingress_reg[3];

    assign mxm_out_00_dbg            = u_lpu.u_mxm.mxm_out[0][0];
    assign mxm_out_01_dbg            = u_lpu.u_mxm.mxm_out[0][1];
    assign mxm_out_02_dbg            = u_lpu.u_mxm.mxm_out[0][2];
    assign mxm_out_03_dbg            = u_lpu.u_mxm.mxm_out[0][3];
    assign mxm_out_10_dbg            = u_lpu.u_mxm.mxm_out[1][0];
    assign mxm_out_11_dbg            = u_lpu.u_mxm.mxm_out[1][1];
    assign mxm_out_12_dbg            = u_lpu.u_mxm.mxm_out[1][2];
    assign mxm_out_13_dbg            = u_lpu.u_mxm.mxm_out[1][3];
    assign mxm_out_20_dbg            = u_lpu.u_mxm.mxm_out[2][0];
    assign mxm_out_21_dbg            = u_lpu.u_mxm.mxm_out[2][1];
    assign mxm_out_22_dbg            = u_lpu.u_mxm.mxm_out[2][2];
    assign mxm_out_23_dbg            = u_lpu.u_mxm.mxm_out[2][3];
    assign mxm_out_30_dbg            = u_lpu.u_mxm.mxm_out[3][0];
    assign mxm_out_31_dbg            = u_lpu.u_mxm.mxm_out[3][1];
    assign mxm_out_32_dbg            = u_lpu.u_mxm.mxm_out[3][2];
    assign mxm_out_33_dbg            = u_lpu.u_mxm.mxm_out[3][3];

`ifdef WAVEFORM
    initial begin
        $dumpfile("lpu_cocotb.vcd");
        $dumpvars(0, lpu_cocotb_top);
    end
`endif

endmodule
