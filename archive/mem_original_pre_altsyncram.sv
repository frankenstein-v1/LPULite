// Archive copy of src/mem.sv before the DE1-SoC HPS FPGA build switched
// MEM0/MEM1 to explicit Intel altsyncram block RAM.
`include "lpu_pkg.sv"

module mem #(
    // Keep this default literal for compatibility with ASIC synthesis
    // frontends that cannot evaluate $bits(typedef) in a parameter list.
    parameter int DATA_W = 72,
    parameter int DEPTH  = MEM_DEPTH,
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  logic clk,
    input  logic rst_n,       // Active-low reset signal

    // Memory rows use the westbound fixed8 format:
    // [63:0] = 8 x int8 fixed-point lanes, [71:64] = shared row scale.
    input  logic [DATA_W-1:0] row_in,
    output logic [DATA_W-1:0] row_out,

    // Control Signals
    input  logic read_en,
    input  logic write_en,
    input  logic [ADDR_W-1:0] addr,

    // External Host/JTAG interface (Bypass ports)
    input  logic              ext_write_en,
    input  logic              ext_read_en,
    input  logic [ADDR_W-1:0] ext_addr,
    input  logic [DATA_W-1:0] ext_data_in,
    output logic [DATA_W-1:0] ext_data_out
);

    // Compatibility names for existing debug/testbench hierarchy references.
    wire [DATA_W-1:0] stream_in;
    logic [DATA_W-1:0] stream_out;

    assign stream_in = row_in;
    assign row_out = stream_out;

`ifdef TINYLPU_USE_SKY130_SRAM
    // The showcase configuration uses 1K rows and a bank of nine 8-bit hard
    // macros. The second macro read port services the optional debug port.
    generate
        if (DATA_W == 72 && DEPTH == 1024) begin : gen_sky130_sram
            tinylpu_sram_banked #(
                .DATA_W(DATA_W)
            ) u_banked_sram (
                .clk,
                .rw_en(read_en || write_en),
                .write_en,
                .rw_addr(addr[9:0]),
                .write_data(row_in),
                .rw_data(stream_out),
                .read2_en(ext_read_en),
                .read2_addr(ext_addr[9:0]),
                .read2_data(ext_data_out)
            );
        end
    endgenerate
`else
    // The actual behavioral SRAM vault. Width defaults to one fixed8 row.
    logic [DATA_W-1:0] sram_array [0:DEPTH-1];

    // Keep writes in a reset-free, single-port process so ASIC synthesis can
    // infer a memory instead of expanding the array into resettable flops.
    always_ff @(posedge clk) begin
        if (write_en) begin
            sram_array[addr] <= stream_in;
        end else if (ext_write_en) begin
            sram_array[ext_addr] <= ext_data_in;
        end
    end

    // SRAM contents are intentionally not reset. Only the registered read
    // outputs are reset; this matches normal compiled-SRAM behavior.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stream_out   <= '0;
            ext_data_out <= '0;
        end else begin
            if (read_en) begin
                stream_out <= sram_array[addr];
            end else begin
                stream_out <= '0;
            end

            if (ext_read_en) begin
                ext_data_out <= sram_array[ext_addr];
            end else begin
                ext_data_out <= '0;
            end
        end
    end
`endif

endmodule
