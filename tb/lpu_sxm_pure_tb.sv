`timescale 1ns/1ps

`include "../src/lpu_pkg.sv"
`include "../src/mac.sv"
`include "../src/acc.sv"
`include "../src/mem.sv"
`include "../src/mxm.sv"
`include "../src/sxm.sv"
`include "../src/icu.sv"
`include "../src/westbound_bus/westbound_bus.sv"
`include "../src/westbound_bus/westbound_consumer_decode.sv"
`include "../src/eastbound_bus/eastbound_bus.sv"
`include "../src/eastbound_bus/eastbound_consumer_decode.sv"
`include "../src/westbound_bus/mxm_eastbound_adapter.sv"
`include "../src/lpu.sv"
`include "LPU_tb.sv"

module lpu_sxm_pure_tb;

    // Clock and Reset
    logic clk;
    logic rst_n;

    // Debug signals
    logic [7:0]  mxm_out_00_dbg;
    logic [31:0] mem0_rdata_dbg;
    logic [31:0] mem1_rdata_dbg;
    logic [31:0] westbound_payload_dbg;
    logic [31:0] eastbound_payload_dbg;
    logic        eastbound_valid_dbg;
    logic        sxm_east_en_dbg;
    logic [31:0] sxm_stream_out_left_dbg;
    logic [31:0] sxm_stream_out_top_dbg;
    logic [31:0] pc_dbg;
    logic [2:0]  westbound_sel_dbg;
    logic [2:0]  westbound_consumer_sel_dbg;
    logic [2:0]  eastbound_sel_dbg;
    logic [2:0]  eastbound_consumer_sel_dbg;
    logic [1:0]  mxm_ingress_mode_dbg;
    logic        mxm_start_dbg;
    logic        mxm_clear_dbg;

    lpu_cocotb_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .mxm_out_00_dbg(mxm_out_00_dbg),
        .westbound_payload_dbg(westbound_payload_dbg),
        .eastbound_payload_dbg(eastbound_payload_dbg),
        .eastbound_valid_dbg(eastbound_valid_dbg),
        .sxm_east_en_dbg(sxm_east_en_dbg),
        .sxm_stream_out_left_dbg(sxm_stream_out_left_dbg),
        .sxm_stream_out_top_dbg(sxm_stream_out_top_dbg)
    );

    // Bind missing debug signals directly from hierarchy
    assign pc_dbg = dut.pc_dbg;
    assign eastbound_sel_dbg = dut.eastbound_sel_dbg;
    assign mxm_start_dbg = dut.mxm_start_dbg;

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Instruction loader task
    task load_inst;
        input int pc;
        input logic [2:0]  westbound_sel;
        input logic [2:0]  eastbound_sel;
        input logic [2:0]  westbound_consumer_sel;
        input logic [2:0]  eastbound_consumer_sel;
        input logic        mem0_read_en;
        input logic        mem0_write_en;
        input logic [8:0]  mem0_addr;
        input logic        mem1_read_en;
        input logic        mem1_write_en;
        input logic [8:0]  mem1_addr;
        input logic [11:0] sxm_opcode_input;
        input logic [11:0] sxm_opcode_weight;
        input logic [1:0]  vxm_math_op;
        input logic        vxm_accum_en;
        input logic        vxm_flush;
        input logic [1:0]  mxm_ingress_mode;
        input logic        mxm_start;
        input logic        mxm_clear;
        input logic [1:0]  mxm_e_row_sel;
        input logic [1:0]  mxm_e_col_sel;
        input logic        mxm_e_valid_in;
        logic [71:0] word;
        begin
            word = 72'd0;
            word[2:0]   = westbound_sel;
            word[5:3]   = eastbound_sel;
            word[8:6]   = westbound_consumer_sel;
            word[11:9]  = eastbound_consumer_sel;
            word[12]    = mem0_read_en;
            word[13]    = mem0_write_en;
            word[22:14] = mem0_addr;
            word[23]    = mem1_read_en;
            word[24]    = mem1_write_en;
            word[33:25] = mem1_addr;
            word[45:34] = sxm_opcode_input;
            word[57:46] = sxm_opcode_weight;
            word[59:58] = vxm_math_op;
            word[60]    = vxm_accum_en;
            word[61]    = vxm_flush;
            word[63:62] = mxm_ingress_mode;
            word[64]    = mxm_start;
            word[65]    = mxm_clear;
            word[67:66] = mxm_e_row_sel;
            word[69:68] = mxm_e_col_sel;
            word[70]    = mxm_e_valid_in;
            dut.u_lpu.u_icu.imem_array[pc] = word;
        end
    endtask

    // Main Test Sequence
    initial begin
        $dumpfile("lpu_sxm_pure.vcd");
        $dumpvars(0, lpu_sxm_pure_tb);

        // Preload memory
        dut.u_lpu.u_mem0.sram_array[0] = 32'h00000002; // Inputs: [2, 0, 0, 0]
        dut.u_lpu.u_mem0.sram_array[1] = 32'h04030201; // Pattern: [4, 3, 2, 1] for SXM test
        dut.u_lpu.u_mem1.sram_array[0] = 32'h00000003; // Weights: [3, 0, 0, 0]

        // Load Instructions
        // PC 0: Load weights
        load_inst(0, 3'd4, 3'd0, 3'd1, 3'd0, 0,0,0, 1,0,0, 0,0, 0,0,0, 2'd2, 0,0, 0,0,0);
        // PC 1: Idle Load
        load_inst(1, 3'd4, 3'd0, 3'd1, 3'd0, 0,0,0, 0,0,0, 0,0, 0,0,0, 2'd2, 0,0, 0,0,0);
        // PC 2: Load inputs
        load_inst(2, 3'd2, 3'd0, 3'd1, 3'd0, 1,0,0, 0,0,0, 0,0, 0,0,0, 2'd1, 0,0, 0,0,0);
        // PC 3: Idle Load
        load_inst(3, 3'd2, 3'd0, 3'd1, 3'd0, 0,0,0, 0,0,0, 0,0, 0,0,0, 2'd1, 0,0, 0,0,0);
        // PC 4: Start MXM
        load_inst(4, 3'd0, 3'd0, 3'd0, 3'd0, 0,0,0, 0,0,0, 0,0, 0,0,0, 2'd0, 1,0, 0,0,0);
        // PC 5: Start MXM
        load_inst(5, 3'd0, 3'd0, 3'd0, 3'd0, 0,0,0, 0,0,0, 0,0, 0,0,0, 2'd0, 1,0, 0,0,0);
        // PC 6: Route Eastbound
        load_inst(6, 3'd0, 3'd1, 3'd0, 3'd1, 0,0,0, 0,0,0, 12'd0,0, 0,0,0, 2'd0, 0,0, 0,0,1);
        // PC 7, 8: Idle
        load_inst(7, 3'd0, 3'd1, 3'd0, 3'd1, 0,0,0, 0,0,0, 12'd0,0, 0,0,0, 2'd0, 0,0, 0,0,1);
        load_inst(8, 3'd0, 3'd0, 3'd0, 3'd0, 0,0,0, 0,0,0, 0,0, 0,0,0, 2'd0, 0,0, 0,0,0);

        // --- EXTENDED SXM FEATURE TESTING ---
        // PC 9: Issue Mem0 Read Addr 1
        load_inst(9,  3'd0, 3'd0, 3'd0, 3'd0, 1,0,9'd1, 0,0,0, 12'd0,0, 0,0,0, 2'd0, 0,0, 0,0,0);
        // PC 10: Route Mem0 to SXM (Loads 04030201 into sxm_e_payload_reg)
        load_inst(10, 3'd0, 3'd3, 3'd0, 3'd1, 0,0,0,    0,0,0, 12'd0,0, 0,0,0, 2'd0, 0,0, 0,0,0);
        // PC 11: Test Crossbar 0 (all lanes get lane 0 = 01)
        load_inst(11, 3'd0, 3'd0, 3'd0, 3'd0, 0,0,0,    0,0,0, 12'b000_000_000_000, 0, 0,0,0, 2'd0, 0,0, 0,0,0);
        // PC 12: Test Crossbar 1 (all lanes get lane 1 = 02). Also Issue Mem0 Read Addr 2 (00000000)
        load_inst(12, 3'd0, 3'd0, 3'd0, 3'd0, 1,0,9'd2, 0,0,0, 12'b001_001_001_001, 0, 0,0,0, 2'd0, 0,0, 0,0,0);
        // PC 13: Route Mem0 to SXM (Loads 0 into sxm_e_payload_reg). Test Bubbles.
        load_inst(13, 3'd0, 3'd3, 3'd0, 3'd1, 0,0,0,    0,0,0, 12'b111_111_111_111, 0, 0,0,0, 2'd0, 0,0, 0,0,0);
        // PC 14: Test Delay 1. (sxm_e_payload is now 0, but input_d1 holds 04030201)
        load_inst(14, 3'd0, 3'd0, 3'd0, 3'd0, 0,0,0,    0,0,0, 12'b100_100_100_100, 0, 0,0,0, 2'd0, 0,0, 0,0,0);
        // PC 15: Idle
        load_inst(15, 3'd0, 3'd0, 3'd0, 3'd0, 0,0,0,    0,0,0, 12'd0,0, 0,0,0, 2'd0, 0,0, 0,0,0);

        // Assert Reset
        rst_n = 0;
        #20;
        rst_n = 1;

        // Wait until PC reaches 6
        wait (pc_dbg == 6);
        #10; // Wait for cycle to complete

        // Verify MAC calculated correctly
        $display("MXM Output: %d", mxm_out_00_dbg);
        if (mxm_out_00_dbg !== 8'd6)
            $display("ERROR: Expected MXM output 6, got %d", mxm_out_00_dbg);

        // Wait until PC reaches 7
        wait (pc_dbg == 7);
        #10;

        // Verify Eastbound and SXM
        $display("Eastbound payload: %d", eastbound_payload_dbg);
        $display("SXM Lane 0 Output: %d", sxm_stream_out_left_dbg[7:0]);
        
        if (sxm_stream_out_left_dbg[7:0] !== 8'd6) begin
            $display("ERROR: Expected SXM output 6, got %d", sxm_stream_out_left_dbg[7:0]);
            $finish;
        end else begin
            $display("SUCCESS: Test passed! MXM output routed to SXM successfully.");
        end

        // Wait until PC reaches 11 (SXM Crossbar 0 test)
        wait (pc_dbg == 11);
        #10;
        if (sxm_stream_out_left_dbg !== 32'h01010101) begin
            $display("ERROR: Crossbar 0 failed. Expected 01010101, got %x", sxm_stream_out_left_dbg);
            $finish;
        end else begin
            $display("SUCCESS: SXM Crossbar 0 passed!");
        end

        // Wait until PC reaches 12 (SXM Crossbar 1 test)
        wait (pc_dbg == 12);
        #10;
        if (sxm_stream_out_left_dbg !== 32'h02020202) begin
            $display("ERROR: Crossbar 1 failed. Expected 02020202, got %x", sxm_stream_out_left_dbg);
            $finish;
        end else begin
            $display("SUCCESS: SXM Crossbar 1 passed!");
        end

        // Wait until PC reaches 13 (SXM Bubbles test)
        wait (pc_dbg == 13);
        #10;
        if (sxm_stream_out_left_dbg !== 32'h00000000) begin
            $display("ERROR: Bubbles failed. Expected 00000000, got %x", sxm_stream_out_left_dbg);
            $finish;
        end else begin
            $display("SUCCESS: SXM Bubbles passed!");
        end

        // Wait until PC reaches 14 (SXM Delay 1 test)
        wait (pc_dbg == 14);
        #10;
        // Delay 1 means it outputs the previous cycle's captured value (04030201)
        if (sxm_stream_out_left_dbg !== 32'h04030201) begin
            $display("ERROR: Delay 1 failed. Expected 04030201, got %x", sxm_stream_out_left_dbg);
            $finish;
        end else begin
            $display("SUCCESS: SXM Delay 1 passed!");
        end

        #50;
        $finish;
    end

endmodule
