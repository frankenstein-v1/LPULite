`include "lpu_pkg.sv"

module mem #(
    parameter int DATA_W = $bits(superlane_t)
) (
    input  logic clk,
    input  logic rst_n,       // Active-low reset signal

    // The Conveyor Belt (Streams)
    input  logic [DATA_W-1:0] stream_in,
    output logic [DATA_W-1:0] stream_out,

    // Control Signals
    input  logic read_en,
    input  logic write_en,
    input  logic [8:0] addr   // 9 bits to address 320 slots (0 to 511)
);

    // The actual SRAM vault. Width is parameterized so LPU can use row-wide storage.
    logic [DATA_W-1:0] sram_array [0:MEM_DEPTH-1];

    // Sequential logic: Everything happens strictly on the clock edge
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // On reset, we clear the outgoing stream.
            // (We deliberately don't clear the whole SRAM array because 
            // wiring a reset to thousands of SRAM cells wastes physical space).
            stream_out <= '0;
        end else begin
            
            // Synchronous Write: Open the SRAM vault and catch incoming data
            if (write_en) begin
                sram_array[addr] <= stream_in;
            end

            // Synchronous Read: Grab data from SRAM and push it onto the stream
            if (read_en) begin
                stream_out <= sram_array[addr];
            end else begin
                // Push bubbles (all zeros) onto the stream if we aren't reading
                stream_out <= '0; 
            end
            
        end
    end

endmodule
