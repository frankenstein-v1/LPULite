`include "lpu_pkg.sv"

module mem #(
    parameter int DATA_W = $bits(superlane_t),
    parameter int DEPTH  = MEM_DEPTH,
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  logic clk,
    input  logic rst_n,       // Active-low reset signal

    // The Conveyor Belt (Streams)
    input  logic [DATA_W-1:0] stream_in,
    output logic [DATA_W-1:0] stream_out,

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

    // The actual SRAM vault. Width is parameterized so LPU can use row-wide storage.
    logic [DATA_W-1:0] sram_array [0:DEPTH-1];

    // Sequential logic: Everything happens strictly on the clock edge
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // On reset, we clear the outgoing stream.
            stream_out   <= '0;
            ext_data_out <= '0;
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

