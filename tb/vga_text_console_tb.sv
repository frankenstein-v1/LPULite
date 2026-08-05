`timescale 1ns/1ps

module vga_text_console_tb;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic write_valid = 1'b0;
    logic [7:0] write_char = " ";
    logic ready;
    logic [7:0] r, g, b;
    logic vga_clk, blank_n, hs, vs;

    always #10 clk = ~clk;

    vga_text_console dut (
        .clk_50(clk), .rst_n, .write_valid, .write_char, .ready,
        .vga_r(r), .vga_g(g), .vga_b(b), .vga_clk,
        .vga_blank_n(blank_n), .vga_hs(hs), .vga_vs(vs)
    );

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        // One clear write is issued per 25 MHz pixel clock (every second
        // system clock); leave extra cycles for registered state updates.
        repeat (5000) @(posedge clk);
        if (!ready) $fatal(1, "console did not finish clearing its text RAM");
        write_char = "A";
        write_valid = 1'b1;
        @(posedge clk);
        write_valid = 1'b0;
        repeat (20) @(posedge clk);
        $display("vga_text_console smoke test passed");
        $finish;
    end
endmodule
