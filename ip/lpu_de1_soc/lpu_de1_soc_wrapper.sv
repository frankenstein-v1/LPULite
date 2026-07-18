`timescale 1ns/1ps
`include "lpu_pkg.sv"

module lpu_de1_soc_wrapper (
    input logic clk, input logic rst_n,
    input logic [15:0] avs_address, input logic avs_read, input logic avs_write,
    input logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata, output logic avs_waitrequest
);
    logic run_enable;
    logic [95:0] imem_assembly;
    logic [71:0] mem0_assembly, mem1_assembly;
    logic ext_imem_write_en;
    logic [9:0] ext_imem_addr;
    logic [95:0] ext_imem_data_in;
    logic ext_mem0_write_en, ext_mem0_read_en, ext_mem1_write_en, ext_mem1_read_en;
    logic [14:0] ext_mem0_addr, ext_mem1_addr;
    logic [71:0] ext_mem0_data_in, ext_mem1_data_in, ext_mem0_data_out, ext_mem1_data_out;
    logic [71:0] mem0_read_latched, mem1_read_latched;

    wire [13:0] imem_word = avs_address[13:2];
    wire [13:0] mem0_word = (avs_address - 16'h4000) >> 2;
    wire [13:0] mem1_word = (avs_address - 16'h8000) >> 2;
    /* Avalon addresses are byte addresses.  Rows use consecutive 32-bit words:
       row = word-address / 3, lane = word-address % 3. */
    integer row_index;
    integer lane_index;

    assign avs_waitrequest = 1'b0;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_enable <= 1'b0; imem_assembly <= '0; mem0_assembly <= '0; mem1_assembly <= '0;
            ext_imem_write_en <= 1'b0; ext_mem0_write_en <= 1'b0; ext_mem1_write_en <= 1'b0;
            ext_mem0_read_en <= 1'b0; ext_mem1_read_en <= 1'b0; avs_readdata <= '0;
            mem0_read_latched <= '0; mem1_read_latched <= '0;
        end else begin
            ext_imem_write_en <= 1'b0; ext_mem0_write_en <= 1'b0; ext_mem1_write_en <= 1'b0;
            ext_mem0_read_en <= 1'b0; ext_mem1_read_en <= 1'b0;
            /* The SRAM read is synchronous.  Capture its result for the read
               transaction launched on the preceding cycle. */
            mem0_read_latched <= ext_mem0_data_out;
            mem1_read_latched <= ext_mem1_data_out;
            if (avs_write) begin
                if (avs_address == 16'hc000) run_enable <= avs_writedata[0];
                else if (avs_address < 16'h4000) begin
                    row_index = imem_word / 3; lane_index = imem_word % 3;
                    case (lane_index)
                      0: imem_assembly[31:0] <= avs_writedata;
                      1: imem_assembly[63:32] <= avs_writedata;
                      2: begin ext_imem_write_en <= 1'b1; ext_imem_addr <= row_index[9:0];
                               ext_imem_data_in <= {avs_writedata, imem_assembly[63:0]}; end
                    endcase
                end else if (avs_address < 16'h8000) begin
                    row_index = mem0_word / 3; lane_index = mem0_word % 3;
                    case (lane_index)
                      0: mem0_assembly[31:0] <= avs_writedata;
                      1: mem0_assembly[63:32] <= avs_writedata;
                      2: begin ext_mem0_write_en <= 1'b1; ext_mem0_addr <= row_index[14:0];
                               ext_mem0_data_in <= {avs_writedata[7:0], mem0_assembly[63:0]}; end
                    endcase
                end else if (avs_address < 16'hc000) begin
                    row_index = mem1_word / 3; lane_index = mem1_word % 3;
                    case (lane_index)
                      0: mem1_assembly[31:0] <= avs_writedata;
                      1: mem1_assembly[63:32] <= avs_writedata;
                      2: begin ext_mem1_write_en <= 1'b1; ext_mem1_addr <= row_index[14:0];
                               ext_mem1_data_in <= {avs_writedata[7:0], mem1_assembly[63:0]}; end
                    endcase
                end
            end
            if (avs_read) begin
                if (avs_address == 16'hc000) avs_readdata <= {31'b0, run_enable};
                else if (avs_address >= 16'h4000 && avs_address < 16'h8000) begin
                    row_index = mem0_word / 3; lane_index = mem0_word % 3;
                    ext_mem0_read_en <= 1'b1; ext_mem0_addr <= row_index[14:0];
                    case (lane_index) 0: avs_readdata <= mem0_read_latched[31:0];
                      1: avs_readdata <= mem0_read_latched[63:32]; default: avs_readdata <= {24'b0,mem0_read_latched[71:64]}; endcase
                end else if (avs_address >= 16'h8000 && avs_address < 16'hc000) begin
                    row_index = mem1_word / 3; lane_index = mem1_word % 3;
                    ext_mem1_read_en <= 1'b1; ext_mem1_addr <= row_index[14:0];
                    case (lane_index) 0: avs_readdata <= mem1_read_latched[31:0];
                      1: avs_readdata <= mem1_read_latched[63:32]; default: avs_readdata <= {24'b0,mem1_read_latched[71:64]}; endcase
                end else avs_readdata <= '0; // IMEM has no external read port.
            end
        end
    end
    lpu u_lpu (.clk(clk), .rst_n(rst_n & run_enable),
        .ext_imem_write_en(ext_imem_write_en), .ext_imem_addr(ext_imem_addr), .ext_imem_data_in(ext_imem_data_in),
        .ext_mem0_write_en(ext_mem0_write_en), .ext_mem0_read_en(ext_mem0_read_en), .ext_mem0_addr(ext_mem0_addr), .ext_mem0_data_in(ext_mem0_data_in), .ext_mem0_data_out(ext_mem0_data_out),
        .ext_mem1_write_en(ext_mem1_write_en), .ext_mem1_read_en(ext_mem1_read_en), .ext_mem1_addr(ext_mem1_addr), .ext_mem1_data_in(ext_mem1_data_in), .ext_mem1_data_out(ext_mem1_data_out));
endmodule
