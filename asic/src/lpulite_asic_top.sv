`timescale 1ns/1ps

// Physical-design boundary for the LPULite compute core.
//
// This wrapper deliberately exposes the existing host programming interface
// and excludes all DE1-SoC/Quartus logic.  The datapath instantiated below is
// the architectural 8x8 MXM, 8-lane VXM configuration from lpu_pkg.sv.
module lpulite_asic_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        run_en,
    input  logic        pc_load_en,
    input  logic [31:0] pc_load_value,
    input  logic        ext_en,
    input  logic        ext_write,
    input  logic [1:0]  ext_target,
    input  logic [31:0] ext_addr,
    input  logic [95:0] ext_wdata,
    output logic [95:0] ext_rdata,
    output logic [31:0] cycle_counter
);

    lpu #(
        .RMSNORM_CHUNKS(8),
        .SOFTMAX_CHUNKS(8),
        // A 1K-row showcase configuration maps cleanly to public SKY130
        // OpenRAM macros while leaving the 8-wide compute architecture intact.
        .DATA_MEM_DEPTH(1024)
    ) u_lpu (
        .clk,
        .rst_n,
        .run_en,
        .pc_load_en,
        .pc_load_value,
        .ext_en,
        .ext_write,
        .ext_target,
        .ext_addr,
        .ext_wdata,
        .ext_rdata,
        .cycle_counter
    );

endmodule
