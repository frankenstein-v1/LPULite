`timescale 1ns/1ps
`include "lpu_pkg.sv"

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
    output logic [2:0]  eastbound_sel_dbg,
    output logic [2:0]  eastbound_consumer_sel_dbg,
    output logic [1:0]  mxm_ingress_mode_dbg,
    output logic        mxm_start_dbg,
    output logic        mxm_clear_dbg,

    output logic        mem0_valid_dbg,
    output logic        mem1_valid_dbg,
    output logic [31:0] westbound_payload_dbg,
    output logic        westbound_valid_dbg,
    output logic        mxm_west_en_dbg,
    output mem_row_t    mem0_stream_in_dbg,
    output logic        mem0_write_from_east_dbg,
    output logic        mem0_write_en_eff_dbg,
    output logic        vxm_input_overflow_dbg,

    output mxm_row_t     eastbound_payload_dbg,
    output logic        eastbound_valid_dbg,
    output logic        sxm_east_en_dbg,
    output logic [31:0] sxm_stream_out_left_dbg,
    output logic [31:0] sxm_stream_out_top_dbg,
    output mxm_row_t    vxm_stream_in_data_dbg,
    output logic        vxm_in_valid_dbg,
    output logic        vxm_in_ready_dbg,
    output logic        vxm_input_fifo_full_dbg,
    output logic        vxm_input_fifo_empty_dbg,
    output mxm_row_t    vxm_scale_row_dbg,
    output logic        vxm_scale_valid_dbg,
    output mxm_row_t    vxm_softmax_in_row_dbg,
    output logic        vxm_softmax_in_valid_dbg,
    output logic [3:0]  vxm_softmax_launch_dbg,
    output logic [3:0]  vxm_softmax_done_dbg,
    output mxm_row_t    vxm_quant_in_row_dbg,
    output logic        vxm_quant_in_valid_dbg,
    output logic [31:0] vxm_stream_out_live_dbg,
    output logic [31:0] vxm_stream_out_scale_live_dbg,
    output logic        vxm_out_valid_live_dbg,
    output logic [31:0] vxm_stream_out_buf_dbg,
    output logic [31:0] vxm_stream_out_scale_buf_dbg,
    output logic        vxm_stream_out_buf_valid_dbg,
    output logic        vxm_output_fifo_full_dbg,
    output logic        vxm_output_fifo_empty_dbg,

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
    output logic signed [31:0] mxm_out_33_dbg,

    output logic        ln_out_valid_dbg,
    output logic [31:0] ln_out_00_dbg,
    output logic [31:0] ln_out_01_dbg,
    output logic [31:0] ln_out_02_dbg,
    output logic [31:0] ln_out_03_dbg,
    output logic [31:0] ln_out_10_dbg,
    output logic [31:0] ln_out_11_dbg,
    output logic [31:0] ln_out_12_dbg,
    output logic [31:0] ln_out_13_dbg,
    output logic [31:0] ln_out_20_dbg,
    output logic [31:0] ln_out_21_dbg,
    output logic [31:0] ln_out_22_dbg,
    output logic [31:0] ln_out_23_dbg,
    output logic [31:0] ln_out_30_dbg,
    output logic [31:0] ln_out_31_dbg,
    output logic [31:0] ln_out_32_dbg,
    output logic [31:0] ln_out_33_dbg,

    input  logic [255:0] ln8_x_dbg,
    input  logic [255:0] ln8_gamma_dbg,
    input  logic [255:0] ln8_beta_dbg,
    output logic [255:0] ln8_y_dbg
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
    assign eastbound_sel_dbg         = u_lpu.eastbound_sel;
    assign eastbound_consumer_sel_dbg = u_lpu.eastbound_consumer_sel;
    assign mxm_ingress_mode_dbg      = u_lpu.mxm_ingress_mode;
    assign mxm_start_dbg             = u_lpu.mxm_start;
    assign mxm_clear_dbg             = u_lpu.mxm_clear;

    assign mem0_valid_dbg            = u_lpu.mem0_valid;
    assign mem1_valid_dbg            = u_lpu.mem1_valid;
    assign westbound_payload_dbg     = u_lpu.westbound_payload;
    assign westbound_valid_dbg       = u_lpu.westbound_valid;
    assign mxm_west_en_dbg           = u_lpu.mxm_west_en;
    assign mem0_stream_in_dbg        = u_lpu.mem0_stream_in;
    assign mem0_write_from_east_dbg  = u_lpu.mem0_write_from_east;
    assign mem0_write_en_eff_dbg     = u_lpu.mem0_write_en_eff;
    assign vxm_input_overflow_dbg    = u_lpu.vxm_input_overflow;

    assign eastbound_payload_dbg     = u_lpu.eastbound_payload;
    assign eastbound_valid_dbg       = u_lpu.eastbound_valid;
    assign sxm_east_en_dbg           = u_lpu.sxm_east_en;
    assign sxm_stream_out_left_dbg   = u_lpu.sxm_stream_out_to_mxm_left;
    assign sxm_stream_out_top_dbg    = u_lpu.sxm_stream_out_to_mxm_top;
    assign vxm_stream_in_data_dbg    = u_lpu.vxm_stream_in_data;
    assign vxm_in_valid_dbg          = u_lpu.vxm_in_valid;
    assign vxm_in_ready_dbg          = u_lpu.vxm_in_ready;
    assign vxm_input_fifo_full_dbg   = u_lpu.vxm_fifo_full;
    assign vxm_input_fifo_empty_dbg  = u_lpu.vxm_fifo_empty;
    assign vxm_scale_row_dbg         = u_lpu.u_vxm.s3_scale_reg;
    assign vxm_scale_valid_dbg       = u_lpu.u_vxm.s3_valid;
    assign vxm_softmax_in_row_dbg    = u_lpu.u_vxm.s4_handoff_reg;
    assign vxm_softmax_in_valid_dbg  = u_lpu.u_vxm.s4_valid;
    assign vxm_softmax_launch_dbg    = u_lpu.u_vxm.softmax_launch_vec;
    assign vxm_softmax_done_dbg      = u_lpu.u_vxm.softmax_valid_vec;
    assign vxm_quant_in_row_dbg      = u_lpu.u_vxm.mux_out;
    assign vxm_quant_in_valid_dbg    = u_lpu.u_vxm.quant_issue;
    assign vxm_stream_out_live_dbg   = u_lpu.vxm_stream_out_live;
    assign vxm_stream_out_scale_live_dbg = u_lpu.vxm_stream_out_scale_live;
    assign vxm_out_valid_live_dbg    = u_lpu.vxm_out_valid_live;
    assign vxm_stream_out_buf_dbg    = u_lpu.vxm_stream_out_buf;
    assign vxm_stream_out_scale_buf_dbg = u_lpu.vxm_stream_out_scale_buf;
    assign vxm_stream_out_buf_valid_dbg = u_lpu.vxm_stream_out_buf_valid_e;
    assign vxm_output_fifo_full_dbg  = u_lpu.vxm_result_full;
    assign vxm_output_fifo_empty_dbg = u_lpu.vxm_result_empty;

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

    lut_layernorm #(
        .LANES(8),
        .LANE_W(32)
    ) u_ln8_dbg (
        .x_in(ln8_x_dbg),
        .gamma(ln8_gamma_dbg),
        .beta(ln8_beta_dbg),
        .y_out(ln8_y_dbg)
    );

`ifdef WAVEFORM
    initial begin
        $dumpfile("lpu_cocotb.vcd");
        $dumpvars(0, lpu_cocotb_top);
    end
`endif

endmodule
