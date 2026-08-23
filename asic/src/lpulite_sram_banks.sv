`timescale 1ns/1ps

// Banks the public 8x1024 SKY130 OpenRAM macro into the widths used by
// LPULite.  Each generated instance is a visible hard macro in the final GDS.
module lpulite_sram_banked #(
    parameter int DATA_W = 72,
    parameter int BANKS  = (DATA_W + 7) / 8
) (
    input  logic              clk,
    input  logic              rw_en,
    input  logic              write_en,
    input  logic [9:0]        rw_addr,
    input  logic [DATA_W-1:0] write_data,
    output logic [DATA_W-1:0] rw_data,
    input  logic              read2_en,
    input  logic [9:0]        read2_addr,
    output logic [DATA_W-1:0] read2_data
);

    logic [BANKS*8-1:0] bank_din;
    logic [BANKS*8-1:0] bank_dout0;
    logic [BANKS*8-1:0] bank_dout1;

    always_comb begin
        bank_din = '0;
        bank_din[DATA_W-1:0] = write_data;
        rw_data = bank_dout0[DATA_W-1:0];
        read2_data = bank_dout1[DATA_W-1:0];
    end

    genvar bank;
    generate
        for (bank = 0; bank < BANKS; bank++) begin : gen_sram_bank
            sky130_sram_1kbyte_1rw1r_8x1024_8 u_sram (
                .clk0(clk),
                .csb0(!rw_en),
                .web0(!write_en),
                .wmask0(1'b1),
                .addr0(rw_addr),
                .din0(bank_din[bank*8 +: 8]),
                .dout0(bank_dout0[bank*8 +: 8]),
                .clk1(clk),
                .csb1(!read2_en),
                .addr1(read2_addr),
                .dout1(bank_dout1[bank*8 +: 8])
            );
        end
    endgenerate

endmodule
