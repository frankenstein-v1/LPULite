`timescale 1ns/1ps
import lpu_pkg::*;

module vxm_tb();

    // 1. The Wires
    logic clk;
    logic rst_n;
    logic [1:0] opcode;
    superlane_t stream_in_data;
    superlane_t stream_in_param;
    superlane_t stream_out;

    // 2. The Device Under Test (DUT)
    vxm dut (
        .clk(clk),
        .rst_n(rst_n),
        .opcode(opcode),
        .stream_in_data(stream_in_data),
        .stream_in_param(stream_in_param),
        .stream_out(stream_out)
    );

    // 3. The Heartbeat (10ns period)
    always #5 clk = ~clk;

    // 4. The Stimulus
    initial begin
        // Tell the simulator to record the waveforms
        $dumpfile("build/vxm.vcd");
        $dumpvars(0, vxm_tb);

        // --- RESET PHASE ---
        clk = 0;
        rst_n = 0;
        opcode = 0;
        stream_in_data = 0;
        stream_in_param = 0;

        #15 rst_n = 1;

        // --- TEST 1: ReLU (Opcode 00) ---
        // Let's test Lane 3 (Positive), Lane 2 (Negative), Lane 1 (Positive), Lane 0 (Negative)
        // 8'hF6 is -10 in two's complement. 8'hFF is -1.
        @(posedge clk);
        opcode = 2'b00;
        stream_in_data = {8'd5, 8'hF6, 8'd15, 8'hFF}; 
        stream_in_param = 32'd0; // ReLU ignores the parameter stream
        
        // --- TEST 2: Bias+ (Opcode 01) ---
        // Adding 1, 2, 3, and 4 to the four lanes respectively
        @(posedge clk);
        opcode = 2'b01;
        stream_in_data = {8'd10, 8'd20, 8'd30, 8'd40};
        stream_in_param = {8'd1, 8'd2, 8'd3, 8'd4};

        // --- TEST 3: Scale (Opcode 10) ---
        // Multiplying the lanes by 10
        @(posedge clk);
        opcode = 2'b10;
        stream_in_data = {8'd2, 8'd3, 8'd4, 8'd5};
        stream_in_param = {8'd10, 8'd10, 8'd10, 8'd10};

        // --- TEST 4: Accumulate (Opcode 11) ---
        // Physically identical to Bias+, just checking the routing works
        @(posedge clk);
        opcode = 2'b11;
        stream_in_data = {8'd100, 8'd50, 8'd25, 8'd12};
        stream_in_param = {8'd5, 8'd5, 8'd5, 8'd5};

        // Flush with zeros to see the final output pop out
        @(posedge clk);
        opcode = 2'b00;
        stream_in_data = 0;
        stream_in_param = 0;

        #30 $finish;
    end

endmodule