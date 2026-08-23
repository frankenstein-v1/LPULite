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

`ifdef LPULITE_USE_SKY130_SRAM
    // The showcase configuration uses 1K rows and a bank of nine 8-bit hard
    // macros. The second macro read port services the optional debug port.
    generate
        if (DATA_W == 72 && DEPTH == 1024) begin : gen_sky130_sram
            lpulite_sram_banked #(
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

`ifdef LPULITE_FPGA_ALTSYNCRAM
    logic [ADDR_W-1:0] ram_addr_a;
    logic [DATA_W-1:0] ram_data_a;
    logic [DATA_W-1:0] ram_q_a;
    logic [DATA_W-1:0] ram_q_b;
    logic              ram_wren_a;
    assign ram_addr_a = ext_write_en ? ext_addr : addr;
    assign ram_data_a = ext_write_en ? ext_data_in : stream_in;
    assign ram_wren_a = write_en || ext_write_en;

    // The block RAM outputs are registered.  Do not gate them back to zero
    // with a delayed read-enable here: consumers already use explicit valid
    // signals, and HPS/JTAG readback samples after the registered latency.
    // Gating the q output can erase the data on the exact cycle the bridge
    // returns it.
    assign stream_out   = ram_q_a;
    assign ext_data_out = ram_q_b;

    altsyncram #(
        .operation_mode("BIDIR_DUAL_PORT"),
        .intended_device_family("Cyclone V"),
        .width_a(DATA_W),
        .widthad_a(ADDR_W),
        .numwords_a(DEPTH),
        .width_b(DATA_W),
        .widthad_b(ADDR_W),
        .numwords_b(DEPTH),
        .outdata_reg_a("CLOCK0"),
        .outdata_reg_b("CLOCK0"),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .wrcontrol_wraddress_reg_b("CLOCK0"),
        .lpm_type("altsyncram"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("FALSE")
    ) u_fpga_ram (
        .clock0(clk),
        .address_a(ram_addr_a),
        .data_a(ram_data_a),
        .wren_a(ram_wren_a),
        .q_a(ram_q_a),
        .address_b(ext_addr),
        .data_b(ext_data_in),
        .wren_b(1'b0),
        .q_b(ram_q_b),
        .aclr0(1'b0),
        .aclr1(1'b0),
        .addressstall_a(1'b0),
        .addressstall_b(1'b0),
        .byteena_a(1'b1),
        .byteena_b(1'b1),
        .clock1(1'b1),
        .clocken0(1'b1),
        .clocken1(1'b1),
        .clocken2(1'b1),
        .clocken3(1'b1),
        .eccstatus()
    );
`else
    // Keep reads synchronous so FPGA synthesis can infer block RAM instead of
    // expanding the array into registers. The integrated LPU routes HPS/JTAG
    // reads through row_out; ext_data_out remains as a synchronous debug
    // readback port for older testbenches.
    always_ff @(posedge clk) begin
        if (write_en) begin
            sram_array[addr] <= stream_in;
        end else if (ext_write_en) begin
            sram_array[ext_addr] <= ext_data_in;
        end

        if (read_en) begin
            stream_out <= sram_array[addr];
        end
    end

    always_ff @(posedge clk) begin
        if (ext_read_en) begin
            ext_data_out <= sram_array[ext_addr];
        end
    end
`endif
`endif

endmodule
