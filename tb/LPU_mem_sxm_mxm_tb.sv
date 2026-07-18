`timescale 1ns/1ps
`include "lpu_pkg.sv"

module lpu_mem_sxm_mxm_cocotb_top (
    input  logic clk,
    input  logic rst_n,

    output logic [31:0] pc_dbg,

    output logic [11:0] sxm_opcode_input_dbg,

    output logic        mem0_read_en_dbg,
    output logic [MEM_ADDR_W-1:0]  mem0_addr_dbg,
    output logic        mem0_valid_dbg,
    output logic [63:0] mem0_stream_out_dbg,

    output logic        mem1_read_en_dbg,
    output logic [MEM_ADDR_W-1:0]  mem1_addr_dbg,
    output logic        mem1_valid_dbg,
    output logic [63:0] mem1_stream_out_dbg,

    output logic [2:0]  westbound_sel_dbg,
    output logic [2:0]  westbound_consumer_sel_dbg,
    output logic [63:0] westbound_payload_dbg,
    output logic        westbound_valid_dbg,

    output logic        sxm_west_en_dbg,
    output logic        sxm_emit_valid_dbg,
    output logic [63:0] sxm_stream_out_left_dbg,
    output logic [63:0] sxm_stream_out_top_dbg,

    output logic [1:0]  mxm_ingress_mode_dbg,
    output logic        mxm_start_dbg,
    output logic        mxm_clear_dbg,
    output logic        mxm_west_en_dbg,

    output logic        input_loaded_dbg,
    output logic signed [7:0] input_buf0,
    output logic signed [7:0] input_buf1,
    output logic signed [7:0] input_buf2,
    output logic signed [7:0] input_buf3,
    output logic signed [7:0] input_buf4,
    output logic signed [7:0] input_buf5,
    output logic signed [7:0] input_buf6,
    output logic signed [7:0] input_buf7,

    output logic        wght_loaded_dbg,
    output logic signed [7:0] wght_buf0,
    output logic signed [7:0] wght_buf1,
    output logic signed [7:0] wght_buf2,
    output logic signed [7:0] wght_buf3,
    output logic signed [7:0] wght_buf4,
    output logic signed [7:0] wght_buf5,
    output logic signed [7:0] wght_buf6,
    output logic signed [7:0] wght_buf7,

    output logic signed [31:0] mxm_out_00_dbg, mxm_out_01_dbg, mxm_out_02_dbg, mxm_out_03_dbg, mxm_out_04_dbg, mxm_out_05_dbg, mxm_out_06_dbg, mxm_out_07_dbg,
    output logic signed [31:0] mxm_out_10_dbg, mxm_out_11_dbg, mxm_out_12_dbg, mxm_out_13_dbg, mxm_out_14_dbg, mxm_out_15_dbg, mxm_out_16_dbg, mxm_out_17_dbg,
    output logic signed [31:0] mxm_out_20_dbg, mxm_out_21_dbg, mxm_out_22_dbg, mxm_out_23_dbg, mxm_out_24_dbg, mxm_out_25_dbg, mxm_out_26_dbg, mxm_out_27_dbg,
    output logic signed [31:0] mxm_out_30_dbg, mxm_out_31_dbg, mxm_out_32_dbg, mxm_out_33_dbg, mxm_out_34_dbg, mxm_out_35_dbg, mxm_out_36_dbg, mxm_out_37_dbg,
    output logic signed [31:0] mxm_out_40_dbg, mxm_out_41_dbg, mxm_out_42_dbg, mxm_out_43_dbg, mxm_out_44_dbg, mxm_out_45_dbg, mxm_out_46_dbg, mxm_out_47_dbg,
    output logic signed [31:0] mxm_out_50_dbg, mxm_out_51_dbg, mxm_out_52_dbg, mxm_out_53_dbg, mxm_out_54_dbg, mxm_out_55_dbg, mxm_out_56_dbg, mxm_out_57_dbg,
    output logic signed [31:0] mxm_out_60_dbg, mxm_out_61_dbg, mxm_out_62_dbg, mxm_out_63_dbg, mxm_out_64_dbg, mxm_out_65_dbg, mxm_out_66_dbg, mxm_out_67_dbg,
    output logic signed [31:0] mxm_out_70_dbg, mxm_out_71_dbg, mxm_out_72_dbg, mxm_out_73_dbg, mxm_out_74_dbg, mxm_out_75_dbg, mxm_out_76_dbg, mxm_out_77_dbg
);

    lpu u_lpu (
        .clk(clk),
        .rst_n(rst_n),
        .run_en(1'b1),
        .pc_load_en(1'b0),
        .pc_load_value(32'd0),
        .ext_en(1'b0),
        .ext_write(1'b0),
        .ext_target(2'd0),
        .ext_addr(32'd0),
        .ext_wdata(96'd0),
        .ext_rdata(),
        .cycle_counter()
    );

    assign pc_dbg                    = u_lpu.u_icu.pc;
    assign sxm_opcode_input_dbg      = u_lpu.sxm_opcode_input;

    assign mem0_read_en_dbg          = u_lpu.mem0_read_en;
    assign mem0_addr_dbg             = u_lpu.mem0_addr;
    assign mem0_valid_dbg            = u_lpu.mem0_valid;
    assign mem0_stream_out_dbg       = fp8_row_mem_packed(u_lpu.mem0_stream_out);

    assign mem1_read_en_dbg          = u_lpu.mem1_read_en;
    assign mem1_addr_dbg             = u_lpu.mem1_addr;
    assign mem1_valid_dbg            = u_lpu.mem1_valid;
    assign mem1_stream_out_dbg       = fp8_row_mem_packed(u_lpu.mem1_stream_out);

    assign westbound_sel_dbg         = u_lpu.westbound_sel;
    assign westbound_consumer_sel_dbg = u_lpu.westbound_consumer_sel;
    assign westbound_payload_dbg     = u_lpu.westbound_payload;
    assign westbound_valid_dbg       = u_lpu.westbound_valid;

    assign sxm_west_en_dbg           = u_lpu.sxm_west_en;
    assign sxm_emit_valid_dbg        = u_lpu.sxm_emit_valid;
    assign sxm_stream_out_left_dbg   = u_lpu.sxm_stream_out_to_mxm_left;
    assign sxm_stream_out_top_dbg    = u_lpu.sxm_stream_out_to_mxm_top;

    assign mxm_ingress_mode_dbg      = u_lpu.mxm_ingress_mode;
    assign mxm_start_dbg             = u_lpu.mxm_start;
    assign mxm_clear_dbg             = u_lpu.mxm_clear;
    assign mxm_west_en_dbg           = u_lpu.mxm_west_en;

    assign input_loaded_dbg          = u_lpu.u_mxm.mxm_input_ingress_loaded;
    assign input_buf0                = u_lpu.u_mxm.mxm_input_ingress_reg[0];
    assign input_buf1                = u_lpu.u_mxm.mxm_input_ingress_reg[1];
    assign input_buf2                = u_lpu.u_mxm.mxm_input_ingress_reg[2];
    assign input_buf3                = u_lpu.u_mxm.mxm_input_ingress_reg[3];
    assign input_buf4                = u_lpu.u_mxm.mxm_input_ingress_reg[4];
    assign input_buf5                = u_lpu.u_mxm.mxm_input_ingress_reg[5];
    assign input_buf6                = u_lpu.u_mxm.mxm_input_ingress_reg[6];
    assign input_buf7                = u_lpu.u_mxm.mxm_input_ingress_reg[7];

    assign wght_loaded_dbg           = u_lpu.u_mxm.mxm_wght_ingress_loaded;
    assign wght_buf0                 = u_lpu.u_mxm.mxm_wght_ingress_reg[0];
    assign wght_buf1                 = u_lpu.u_mxm.mxm_wght_ingress_reg[1];
    assign wght_buf2                 = u_lpu.u_mxm.mxm_wght_ingress_reg[2];
    assign wght_buf3                 = u_lpu.u_mxm.mxm_wght_ingress_reg[3];
    assign wght_buf4                 = u_lpu.u_mxm.mxm_wght_ingress_reg[4];
    assign wght_buf5                 = u_lpu.u_mxm.mxm_wght_ingress_reg[5];
    assign wght_buf6                 = u_lpu.u_mxm.mxm_wght_ingress_reg[6];
    assign wght_buf7                 = u_lpu.u_mxm.mxm_wght_ingress_reg[7];

    assign mxm_out_00_dbg            = u_lpu.u_mxm.mxm_out[0][0];
    assign mxm_out_01_dbg            = u_lpu.u_mxm.mxm_out[0][1];
    assign mxm_out_02_dbg            = u_lpu.u_mxm.mxm_out[0][2];
    assign mxm_out_03_dbg            = u_lpu.u_mxm.mxm_out[0][3];
    assign mxm_out_04_dbg            = u_lpu.u_mxm.mxm_out[0][4];
    assign mxm_out_05_dbg            = u_lpu.u_mxm.mxm_out[0][5];
    assign mxm_out_06_dbg            = u_lpu.u_mxm.mxm_out[0][6];
    assign mxm_out_07_dbg            = u_lpu.u_mxm.mxm_out[0][7];

    assign mxm_out_10_dbg            = u_lpu.u_mxm.mxm_out[1][0];
    assign mxm_out_11_dbg            = u_lpu.u_mxm.mxm_out[1][1];
    assign mxm_out_12_dbg            = u_lpu.u_mxm.mxm_out[1][2];
    assign mxm_out_13_dbg            = u_lpu.u_mxm.mxm_out[1][3];
    assign mxm_out_14_dbg            = u_lpu.u_mxm.mxm_out[1][4];
    assign mxm_out_15_dbg            = u_lpu.u_mxm.mxm_out[1][5];
    assign mxm_out_16_dbg            = u_lpu.u_mxm.mxm_out[1][6];
    assign mxm_out_17_dbg            = u_lpu.u_mxm.mxm_out[1][7];

    assign mxm_out_20_dbg            = u_lpu.u_mxm.mxm_out[2][0];
    assign mxm_out_21_dbg            = u_lpu.u_mxm.mxm_out[2][1];
    assign mxm_out_22_dbg            = u_lpu.u_mxm.mxm_out[2][2];
    assign mxm_out_23_dbg            = u_lpu.u_mxm.mxm_out[2][3];
    assign mxm_out_24_dbg            = u_lpu.u_mxm.mxm_out[2][4];
    assign mxm_out_25_dbg            = u_lpu.u_mxm.mxm_out[2][5];
    assign mxm_out_26_dbg            = u_lpu.u_mxm.mxm_out[2][6];
    assign mxm_out_27_dbg            = u_lpu.u_mxm.mxm_out[2][7];

    assign mxm_out_30_dbg            = u_lpu.u_mxm.mxm_out[3][0];
    assign mxm_out_31_dbg            = u_lpu.u_mxm.mxm_out[3][1];
    assign mxm_out_32_dbg            = u_lpu.u_mxm.mxm_out[3][2];
    assign mxm_out_33_dbg            = u_lpu.u_mxm.mxm_out[3][3];
    assign mxm_out_34_dbg            = u_lpu.u_mxm.mxm_out[3][4];
    assign mxm_out_35_dbg            = u_lpu.u_mxm.mxm_out[3][5];
    assign mxm_out_36_dbg            = u_lpu.u_mxm.mxm_out[3][6];
    assign mxm_out_37_dbg            = u_lpu.u_mxm.mxm_out[3][7];

    assign mxm_out_40_dbg            = u_lpu.u_mxm.mxm_out[4][0];
    assign mxm_out_41_dbg            = u_lpu.u_mxm.mxm_out[4][1];
    assign mxm_out_42_dbg            = u_lpu.u_mxm.mxm_out[4][2];
    assign mxm_out_43_dbg            = u_lpu.u_mxm.mxm_out[4][3];
    assign mxm_out_44_dbg            = u_lpu.u_mxm.mxm_out[4][4];
    assign mxm_out_45_dbg            = u_lpu.u_mxm.mxm_out[4][5];
    assign mxm_out_46_dbg            = u_lpu.u_mxm.mxm_out[4][6];
    assign mxm_out_47_dbg            = u_lpu.u_mxm.mxm_out[4][7];

    assign mxm_out_50_dbg            = u_lpu.u_mxm.mxm_out[5][0];
    assign mxm_out_51_dbg            = u_lpu.u_mxm.mxm_out[5][1];
    assign mxm_out_52_dbg            = u_lpu.u_mxm.mxm_out[5][2];
    assign mxm_out_53_dbg            = u_lpu.u_mxm.mxm_out[5][3];
    assign mxm_out_54_dbg            = u_lpu.u_mxm.mxm_out[5][4];
    assign mxm_out_55_dbg            = u_lpu.u_mxm.mxm_out[5][5];
    assign mxm_out_56_dbg            = u_lpu.u_mxm.mxm_out[5][6];
    assign mxm_out_57_dbg            = u_lpu.u_mxm.mxm_out[5][7];

    assign mxm_out_60_dbg            = u_lpu.u_mxm.mxm_out[6][0];
    assign mxm_out_61_dbg            = u_lpu.u_mxm.mxm_out[6][1];
    assign mxm_out_62_dbg            = u_lpu.u_mxm.mxm_out[6][2];
    assign mxm_out_63_dbg            = u_lpu.u_mxm.mxm_out[6][3];
    assign mxm_out_64_dbg            = u_lpu.u_mxm.mxm_out[6][4];
    assign mxm_out_65_dbg            = u_lpu.u_mxm.mxm_out[6][5];
    assign mxm_out_66_dbg            = u_lpu.u_mxm.mxm_out[6][6];
    assign mxm_out_67_dbg            = u_lpu.u_mxm.mxm_out[6][7];

    assign mxm_out_70_dbg            = u_lpu.u_mxm.mxm_out[7][0];
    assign mxm_out_71_dbg            = u_lpu.u_mxm.mxm_out[7][1];
    assign mxm_out_72_dbg            = u_lpu.u_mxm.mxm_out[7][2];
    assign mxm_out_73_dbg            = u_lpu.u_mxm.mxm_out[7][3];
    assign mxm_out_74_dbg            = u_lpu.u_mxm.mxm_out[7][4];
    assign mxm_out_75_dbg            = u_lpu.u_mxm.mxm_out[7][5];
    assign mxm_out_76_dbg            = u_lpu.u_mxm.mxm_out[7][6];
    assign mxm_out_77_dbg            = u_lpu.u_mxm.mxm_out[7][7];

`ifdef WAVEFORM
    initial begin
        $dumpfile("lpu_mem_sxm_mxm_cocotb.vcd");
        $dumpvars(0, lpu_mem_sxm_mxm_cocotb_top);
    end
`endif

endmodule
