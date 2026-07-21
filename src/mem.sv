`include "lpu_pkg.sv"

module mem #(
    parameter int DATA_W = $bits(mem_row_t),
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

    // The actual SRAM vault. Width defaults to one fixed8 memory row.
    logic [DATA_W-1:0] sram_array [0:DEPTH-1];

    // Sequential logic: Everything happens strictly on the clock edge
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // On reset, we clear the outgoing stream.
            stream_out   <= '0;
            ext_data_out <= '0;
            // External writes remain available while reset is asserted.
            if (ext_write_en) sram_array[ext_addr] <= ext_data_in;
        end else begin
            
            // Synchronous Write: Open the SRAM vault and catch incoming data
            if (write_en) begin
                sram_array[addr] <= stream_in;
            end else if (ext_write_en) begin
                sram_array[ext_addr] <= ext_data_in;
            end

            // Synchronous Read: Grab data from SRAM and push it onto the stream
            if (read_en) begin
                stream_out <= sram_array[addr];
            end else begin
                // Push bubbles (all zeros) onto the stream if we aren't reading
                stream_out <= '0; 
            end

            // Synchronous External Read: Grab data for the JTAG host readout
            if (ext_read_en) begin
                ext_data_out <= sram_array[ext_addr];
            end else begin
                ext_data_out <= '0;
            end
            
        end
    end

endmodule
