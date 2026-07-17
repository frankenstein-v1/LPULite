module icu #(
    // How many rows of code your chip's instruction memory can hold
    parameter INSTRUCTION_COUNT = 1024 
)(
    input  logic clk,
    input  logic rst_n,

    // mem0 Control
    output logic      mem0_read_en,
    output logic      mem0_write_en,
    output logic [MEM_ADDR_W-1:0] mem0_addr,

    // mem1 Control
    output logic      mem1_read_en,
    output logic      mem1_write_en,
    output logic [MEM_ADDR_W-1:0] mem1_addr,

    // SXM Control (24 bits)
    output logic [11:0] sxm_opcode_input,
    output logic [11:0] sxm_opcode_weight,

    // VXM Control (4 bits)
    output logic [3:0] vxm_ctrl,
    output logic       vxm_data_sel,
    output logic [2:0] vxm_operand_sel,

    // bus control (12 bits)
    output logic [2:0] westbound_sel,
    output logic [2:0] eastbound_sel,
    output logic [2:0] westbound_consumer_sel,
    output logic [2:0] eastbound_consumer_sel,

    // mxm control (11 bits)
    output logic [1:0] mxm_ingress_mode,
    output logic      mxm_start,
    output logic      mxm_clear,
    output logic [2:0] mxm_e_row_sel,
    output logic [2:0] mxm_e_col_sel,
    output logic      mxm_e_valid_in,
    output logic      mxm_input_is_signed,
    output logic      mxm_wght_is_signed,
    output logic      mxm_use_fp,
    output logic      fp_quant_mode,
    output logic [1:0] mem_store_fmt,
    output logic      vxm_rmsnorm_en,
    output logic      vxm_rope_en,
    output logic [2:0] vxm_residual_op
);

    // 1. THE INSTRUCTION MEMORY (IMEM)
    // This is the physical SRAM vault holding your compiled software.
    // It is 96 bits wide and has INSTRUCTION_COUNT rows.
    logic [95:0] imem_array [0:INSTRUCTION_COUNT-1];

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
    // We grab the current 96-bit word sitting at the current PC address.
    logic [95:0] current_instruction;
    assign current_instruction = imem_array[pc];

    // bus control
    assign westbound_sel          = current_instruction[2:0];
    assign eastbound_sel          = current_instruction[5:3];
    assign westbound_consumer_sel = current_instruction[8:6];
    assign eastbound_consumer_sel = current_instruction[11:9];

    // mem0 control
    assign mem0_read_en  = current_instruction[12];
    assign mem0_write_en = current_instruction[13];
    // Keep the original low 9 address bits in place for backward-compatible
    // programs. The current 96-bit instruction format carries 11 memory-address
    // bits; the new upper address bits are zero until the instruction format is
    // widened or remapped.
    assign mem0_addr     = {{(MEM_ADDR_W-11){1'b0}}, current_instruction[91:90], current_instruction[22:14]};

    // mem1 control
    assign mem1_read_en  = current_instruction[23];
    assign mem1_write_en = current_instruction[24];
    assign mem1_addr     = {{(MEM_ADDR_W-11){1'b0}}, current_instruction[93:92], current_instruction[33:25]};

    // SXM control
    assign sxm_opcode_input  = current_instruction[45:34];
    assign sxm_opcode_weight = current_instruction[57:46];
    
    // VXM control
    assign vxm_ctrl      = current_instruction[74:71];
    assign vxm_data_sel  = current_instruction[76];

    // MXM control
    assign mxm_ingress_mode = current_instruction[63:62];
    assign mxm_start        = current_instruction[64];
    assign mxm_clear        = current_instruction[65];
    assign mxm_e_row_sel    = current_instruction[88:86];
    assign mxm_e_col_sel    = current_instruction[91:89];
    assign mxm_e_valid_in   = current_instruction[92];
    assign mxm_input_is_signed = current_instruction[77];
    assign mxm_wght_is_signed  = current_instruction[78];
    assign mxm_use_fp          = current_instruction[79];
    assign fp_quant_mode       = current_instruction[80];
    assign mem_store_fmt       = current_instruction[82:81];
    assign vxm_rmsnorm_en      = current_instruction[83];
    assign vxm_operand_sel     = current_instruction[86:84];
    assign vxm_rope_en         = current_instruction[87];
    assign vxm_residual_op     = current_instruction[90:88];

endmodule
