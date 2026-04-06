module icu #(
    // How many rows of code your chip's instruction memory can hold
    parameter INSTRUCTION_COUNT = 1024 
)(
    input  logic clk,
    input  logic rst_n,

    // --- OUTGOING CONTROL CABLES ---

    // West MEM Control (11 bits)
    output logic       west_mem_read_en,
    output logic       west_mem_write_en,
    output logic [8:0] west_mem_addr,

    // East MEM Control (11 bits)
    output logic       east_mem_read_en,
    output logic       east_mem_write_en,
    output logic [8:0] east_mem_addr,

    // SXM Control (24 bits)
    output logic [11:0] sxm_opcode_data,
    output logic [11:0] sxm_opcode_weight,

    // MXM Control (6 bits)
    output logic       mxm_clear,
    output logic       mxm_start,
    output logic [3:0] mxm_wght_load,

    // VXM Control (4 bits)
    output logic [1:0] vxm_math_op,
    output logic       vxm_accum_en,
    output logic       vxm_flush
);

    // 1. THE INSTRUCTION MEMORY (IMEM)
    // This is the physical SRAM vault holding your compiled software.
    // It is 64 bits wide and has INSTRUCTION_COUNT rows.
    logic [63:0] imem_array [0:INSTRUCTION_COUNT-1];

    // 2. THE PROGRAM COUNTER (PC)
    // A simple register that tracks what line of code we are on.
    logic [31:0] pc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 32'd0; // Reset back to line 0
        end else begin
            // In a stateless LPU, we never stall or branch. We just relentlessly march forward.
            pc <= pc + 1; 
        end
    end

    // 3. THE SLICER (The Dispatcher)
    // We grab the massive 64-bit word sitting at the current PC address...
    logic [63:0] current_instruction;
    assign current_instruction = imem_array[pc];

    // ...and we physically slice it up and attach it to the output cables.
    // (Bits [63:56] are left unconnected as padding for future upgrades)

    // West MEM [55:45]
    assign west_mem_read_en  = current_instruction[55];
    assign west_mem_write_en = current_instruction[54];
    assign west_mem_addr     = current_instruction[53:45];

    // East MEM [44:34]
    assign east_mem_read_en  = current_instruction[44];
    assign east_mem_write_en = current_instruction[43];
    assign east_mem_addr     = current_instruction[42:34];

    // SXM [33:10]
    assign sxm_opcode_data   = current_instruction[33:22];
    assign sxm_opcode_weight = current_instruction[21:10];

    // MXM [9:4]
    assign mxm_clear         = current_instruction[9];
    assign mxm_start         = current_instruction[8];
    assign mxm_wght_load     = current_instruction[7:4];

    // VXM [3:0]
    assign vxm_math_op       = current_instruction[3:2];
    assign vxm_accum_en      = current_instruction[1];
    assign vxm_flush         = current_instruction[0];

endmodule