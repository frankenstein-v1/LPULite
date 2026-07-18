`timescale 1ns/1ps
`include "lpu_pkg.sv"

module lpu_de1_soc_wrapper (
    input  logic              clk,
    input  logic              rst_n,      // Hardware global reset (active-low)

    // Intel Avalon Memory-Mapped Slave Interface ports
    input  logic [15:0]       avs_address,   // Byte address bus
    input  logic              avs_read,
    input  logic              avs_write,
    input  logic [31:0]       avs_writedata,
    output logic [31:0]       avs_readdata,
    output logic              avs_waitrequest
);

    // Control Register
    // Bit 0: run_rst_n (Active-high run enable; when 0, u_lpu is held in reset for programming)
    logic run_rst_n;

    // Buffer registers for constructing/deconstructing wide memory words
    logic [95:0] imem_buf;
    logic [71:0] mem0_buf;
    logic [71:0] mem1_buf;

    // External interface wires to connect to the LPU instance
    logic        ext_imem_write_en;
    logic [9:0]  ext_imem_addr;
    logic [95:0] ext_imem_data_in;

    logic        ext_mem0_write_en;
    logic        ext_mem0_read_en;
    logic [14:0] ext_mem0_addr;
    logic [71:0] ext_mem0_data_in;
    logic [71:0] ext_mem0_data_out;

    logic        ext_mem1_write_en;
    logic        ext_mem1_read_en;
    logic [14:0] ext_mem1_addr;
    logic [71:0] ext_mem1_data_in;
    logic [71:0] ext_mem1_data_out;

    // Default status wires
    assign avs_waitrequest = 1'b0; // Zero wait-states since all registers are ready in 1 cycle

    // Address Decoding (Byte addresses, word aligned)
    // 0x0000 - 0x3FFF: IMEM (16KB space). 1024 insts * 12 bytes.
    // 0x4000 - 0x7FFF: MEM0 (16KB space). 1024 rows * 12 bytes.
    // 0x8000 - 0xBFFF: MEM1 (16KB space). 1024 rows * 12 bytes.
    // 0xC000         : Control / Status Register (offset 0xC000)

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_rst_n         <= 1'b0; // Default: Halted (reset active internally to LPU)
            imem_buf          <= '0;
            mem0_buf          <= '0;
            mem1_buf          <= '0;
            ext_imem_write_en <= 1'b0;
            ext_imem_addr     <= '0;
            ext_imem_data_in  <= '0;
            ext_mem0_write_en <= 1'b0;
            ext_mem0_read_en  <= 1'b0;
            ext_mem0_addr     <= '0;
            ext_mem0_data_in  <= '0;
            ext_mem1_write_en <= 1'b0;
            ext_mem1_read_en  <= 1'b0;
            ext_mem1_addr     <= '0;
            ext_mem1_data_in  <= '0;
            avs_readdata      <= '0;
        end else begin
            // Defaults
            ext_imem_write_en <= 1'b0;
            ext_mem0_write_en <= 1'b0;
            ext_mem0_read_en  <= 1'b0;
            ext_mem1_write_en <= 1'b0;
            ext_mem1_read_en  <= 1'b0;

            // Handle Avalon-MM Writes
            if (avs_write) begin
                if (avs_address == 16'hC000) begin
                    run_rst_n <= avs_writedata[0];
                end
                
                // Write IMEM Word (96 bits via three consecutive 32-bit registers)
                else if (avs_address >= 16'h0000 && avs_address < 16'h4000) begin
                    logic [9:0]  inst_idx;
                    logic [1:0]  word_select;
                    inst_idx    = avs_address[13:4]; // 12-byte offset: divide by 12 (approx. by shifting)
                    // Let's use simple instruction mapping: address [3:2] determines the 32-bit word select
                    word_select = avs_address[3:2]; 
                    
                    case (word_select)
                        2'b00: imem_buf[31:0]  <= avs_writedata;
                        2'b01: imem_buf[63:32] <= avs_writedata;
                        2'b10: begin
                            imem_buf[95:64]    <= avs_writedata;
                            // Commit the write when the final word (word 2) is written
                            ext_imem_write_en  <= 1'b1;
                            ext_imem_addr      <= avs_address >> 4; // Address shift to row index
                            ext_imem_data_in   <= {avs_writedata, imem_buf[63:0]};
                        end
                        default: ;
                    endcase
                end

                // Write MEM0 Row (72 bits via three consecutive 32-bit registers)
                else if (avs_address >= 16'h4000 && avs_address < 16'h8000) begin
                    logic [1:0] word_select;
                    word_select = avs_address[3:2];
                    case (word_select)
                        2'b00: mem0_buf[31:0]  <= avs_writedata;
                        2'b01: mem0_buf[63:32] <= avs_writedata;
                        2'b10: begin
                            mem0_buf[71:64]    <= avs_writedata[7:0];
                            ext_mem0_write_en  <= 1'b1;
                            ext_mem0_addr      <= (avs_address - 16'h4000) >> 4;
                            ext_mem0_data_in   <= {avs_writedata[7:0], mem0_buf[63:0]};
                        end
                        default: ;
                    endcase
                end

                // Write MEM1 Row (72 bits via three consecutive 32-bit registers)
                else if (avs_address >= 16'h8000 && avs_address < 16'hC000) begin
                    logic [1:0] word_select;
                    word_select = avs_address[3:2];
                    case (word_select)
                        2'b00: mem1_buf[31:0]  <= avs_writedata;
                        2'b01: mem1_buf[63:32] <= avs_writedata;
                        2'b10: begin
                            mem1_buf[71:64]    <= avs_writedata[7:0];
                            ext_mem1_write_en  <= 1'b1;
                            ext_mem1_addr      <= (avs_address - 16'h8000) >> 4;
                            ext_mem1_data_in   <= {avs_writedata[7:0], mem1_buf[63:0]};
                        end
                        default: ;
                    endcase
                end
            end

            // Handle Avalon-MM Reads
            if (avs_read) begin
                if (avs_address == 16'hC000) begin
                    avs_readdata <= {31'b0, run_rst_n};
                end
                
                // Read MEM0 Row (Read registers sequentially)
                else if (avs_address >= 16'h4000 && avs_address < 16'h8000) begin
                    logic [1:0] word_select;
                    word_select = avs_address[3:2];
                    ext_mem0_read_en <= 1'b1;
                    ext_mem0_addr    <= (avs_address - 16'h4000) >> 4;
                    
                    case (word_select)
                        2'b00: avs_readdata <= ext_mem0_data_out[31:0];
                        2'b01: avs_readdata <= ext_mem0_data_out[63:32];
                        2'b10: avs_readdata <= {24'b0, ext_mem0_data_out[71:64]};
                        default: avs_readdata <= 32'b0;
                    endcase
                end

                // Read MEM1 Row (Read registers sequentially)
                else if (avs_address >= 16'h8000 && avs_address < 16'hC000) begin
                    logic [1:0] word_select;
                    word_select = avs_address[3:2];
                    ext_mem1_read_en <= 1'b1;
                    ext_mem1_addr    <= (avs_address - 16'h8000) >> 4;
                    
                    case (word_select)
                        2'b00: avs_readdata <= ext_mem1_data_out[31:0];
                        2'b01: avs_readdata <= ext_mem1_data_out[63:32];
                        2'b10: avs_readdata <= {24'b0, ext_mem1_data_out[71:64]};
                        default: avs_readdata <= 32'b0;
                    endcase
                end
                
                else begin
                    avs_readdata <= 32'b0;
                end
            end
        end
    end

    // Instantiate your hardware core
    // When run_rst_n is low, LPU is held in reset (programming code/memory).
    // When run_rst_n is high, LPU executes normally.
    lpu u_lpu (
        .clk              (clk),
        .rst_n            (rst_n & run_rst_n), // Compound reset: halts LPU when run_rst_n is low

        // IMEM external loader wires
        .ext_imem_write_en(ext_imem_write_en),
        .ext_imem_addr    (ext_imem_addr),
        .ext_imem_data_in (ext_imem_data_in),

        // MEM0 external driver wires
        .ext_mem0_write_en(ext_mem0_write_en),
        .ext_mem0_read_en (ext_mem0_read_en),
        .ext_mem0_addr    (ext_mem0_addr),
        .ext_mem0_data_in (ext_mem0_data_in),
        .ext_mem0_data_out(ext_mem0_data_out),

        // MEM1 external driver wires
        .ext_mem1_write_en(ext_mem1_write_en),
        .ext_mem1_read_en (ext_mem1_read_en),
        .ext_mem1_addr    (ext_mem1_addr),
        .ext_mem1_data_in (ext_mem1_data_in),
        .ext_mem1_data_out(ext_mem1_data_out)
    );

endmodule
