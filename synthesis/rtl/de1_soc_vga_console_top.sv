`timescale 1ns/1ps

// Standalone VGA bring-up image. It is intentionally independent of the LPU
// and JTAG bridge so a monitor/cable/pin problem cannot be confused with a
// model-inference problem.
module de1_soc_vga_console_top (
    input  logic       CLOCK_50,
    input  logic [0:0] KEY,
    output logic [0:0] LEDR,
    output logic [7:0] VGA_R,
    output logic [7:0] VGA_G,
    output logic [7:0] VGA_B,
    output logic       VGA_CLK,
    output logic       VGA_BLANK_N,
    output logic       VGA_HS,
    output logic       VGA_VS,
    output logic       VGA_SYNC_N
);
    logic console_ready;
    logic write_valid;
    logic [7:0] write_char;

    vga_text_console u_console (
        .clk_50(CLOCK_50), .rst_n(KEY[0]),
        .write_valid, .write_char, .ready(console_ready),
        .vga_r(VGA_R), .vga_g(VGA_G), .vga_b(VGA_B),
        .vga_clk(VGA_CLK), .vga_blank_n(VGA_BLANK_N),
        .vga_hs(VGA_HS), .vga_vs(VGA_VS)
    );

    vga_boot_banner u_banner (
        .clk(CLOCK_50), .rst_n(KEY[0]), .console_ready,
        .write_valid, .write_char
    );

    assign LEDR[0] = console_ready;
    // The monitor uses separate H/V synchronization.  Keep the unused
    // composite-sync input asserted, matching the DE1-SoC reference VGA use.
    assign VGA_SYNC_N = 1'b0;
endmodule
