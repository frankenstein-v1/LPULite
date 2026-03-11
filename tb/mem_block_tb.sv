`timescale 1ns/1ps
import lpu_pkg::*;

module mem_block_tb();

    // 1. The Wires (Fake motherboard traces)
    logic clk;
    logic rst_n;
    superlane_t stream_in;
    superlane_t stream_out;
    logic read_en;
    logic write_en;
    logic [8:0] addr;

    // 2. The Device Under Test (DUT)
    // We plug our wires into the ports of the mem_block module
    mem_block dut (
        .clk(clk),
        .rst_n(rst_n),
        .stream_in(stream_in),
        .stream_out(stream_out),
        .read_en(read_en),
        .write_en(write_en),
        .addr(addr)
    );

    // 3. The Heartbeat
    // Toggle the clock every 5 time units (10 unit period)
    always #5 clk = ~clk;

    // 4. The Stimulus (The actual test script)
    initial begin
        // Tell the simulator to record the waveforms for GTKWave
        $dumpfile("build/mem_block.vcd");
        $dumpvars(0, mem_block_tb);

        // --- THE BIG BANG ---
        // Start everything at zero
        clk = 0;
        rst_n = 0; // Active low, so this resets the module
        stream_in = 0;
        read_en = 0;
        write_en = 0;
        addr = 0;

        // Wait a bit, then release the reset
        #15 rst_n = 1;

        // --- TEST 1: Write Data ---
        // Wait for the next clock edge
        @(posedge clk); 
        write_en = 1;
        addr = 9'd42; // Let's write to mailbox 42
        stream_in = 32'hDEADBEEF; // A highly recognizable hex string
        
        // Turn off the write signal on the next clock
        @(posedge clk);
        write_en = 0;
        stream_in = 0;

        // --- TEST 2: Read Data ---
        #20; // Wait a few cycles to prove the memory holds it
        @(posedge clk);
        read_en = 1;
        addr = 9'd42; // Read from mailbox 42
        
        // Turn off read
        @(posedge clk);
        read_en = 0;

        // Let the simulation run for a few more cycles to see the output, then kill it
        #30 $finish;
    end

endmodule