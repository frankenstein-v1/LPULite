module icu #(
    // How many rows of code your chip's instruction memory can hold
    parameter INSTRUCTION_COUNT = 1024 
)(
    input  logic clk,
    input  logic rst_n,
    input  logic run_en,
    input  logic pc_load_en,
    input  logic [31:0] pc_load_value,

    input  logic        ext_imem_en,
    input  logic        ext_imem_write,
    input  logic [9:0]  ext_imem_addr,
    input  logic [95:0] ext_imem_wdata,
    output logic [95:0] ext_imem_rdata,

    // mem0 Control
    output logic      mem0_read_en,
    output logic      mem0_write_en,
    output logic [MEM_ADDR_W-1:0] mem0_addr,

    // mem1 Control
    output logic      mem1_read_en,
    output logic      mem1_write_en,
    output logic [MEM_ADDR_W-1:0] mem1_addr,

    // SXM Control. The 96-bit VLIW only carries transpose controls; these
    // legacy opcode outputs are generated here for the current SXM interface.
    output logic [11:0] sxm_opcode_input,
    output logic [11:0] sxm_opcode_weight,
    output logic        sxm_load_from_west,

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
    output logic [1:0] mem_store_fmt,
    output logic      vxm_rmsnorm_en,
    output logic      vxm_rope_en,
    output logic      vxm_rope_lut_en,
    output logic [7:0] vxm_rope_position,
    output logic [2:0] vxm_residual_op
);

`ifndef LPULITE_USE_SKY130_SRAM
    // Behavioral instruction memory used by simulation and FPGA builds.
    logic [95:0] imem_array [0:INSTRUCTION_COUNT-1];
`endif

    // 2. THE PROGRAM COUNTER (PC)
    // A simple register that tracks what line of code we are on.
    logic [31:0] pc;
    logic [95:0] current_instruction;

`ifdef LPULITE_USE_SKY130_SRAM
    lpulite_sram_banked #(
        .DATA_W(96)
    ) u_imem_sram (
        .clk,
        .rw_en(ext_imem_en),
        .write_en(ext_imem_write),
        .rw_addr(ext_imem_addr),
        .write_data(ext_imem_wdata),
        .rw_data(ext_imem_rdata),
        .read2_en(run_en || pc_load_en),
        .read2_addr(pc[9:0]),
        .read2_data(current_instruction)
    );
`else
`ifdef LPULITE_FPGA_ALTSYNCRAM
    logic [95:0] imem_q_fetch;
    logic [95:0] imem_q_ext;
    logic        ext_imem_read_q;

    always_ff @(posedge clk) begin
        ext_imem_read_q <= ext_imem_en && !ext_imem_write;
    end

    assign current_instruction = imem_q_fetch;
    assign ext_imem_rdata = imem_q_ext;

    altsyncram #(
        .operation_mode("BIDIR_DUAL_PORT"),
        .intended_device_family("Cyclone V"),
        .width_a(96),
        .widthad_a($clog2(INSTRUCTION_COUNT)),
        .numwords_a(INSTRUCTION_COUNT),
        .width_b(96),
        .widthad_b($clog2(INSTRUCTION_COUNT)),
        .numwords_b(INSTRUCTION_COUNT),
        .outdata_reg_a("CLOCK0"),
        .outdata_reg_b("CLOCK0"),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .wrcontrol_wraddress_reg_b("CLOCK0"),
        .lpm_type("altsyncram"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("FALSE")
    ) u_fpga_imem (
        .clock0(clk),
        .address_a(pc[$clog2(INSTRUCTION_COUNT)-1:0]),
        .data_a(96'b0),
        .wren_a(1'b0),
        .q_a(imem_q_fetch),
        .address_b(ext_imem_addr[$clog2(INSTRUCTION_COUNT)-1:0]),
        .data_b(ext_imem_wdata),
        .wren_b(ext_imem_en && ext_imem_write),
        .q_b(imem_q_ext),
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
    always_ff @(posedge clk) begin
        if (ext_imem_en && ext_imem_write) begin
            imem_array[ext_imem_addr] <= ext_imem_wdata;
        end

        // 3. THE SLICER (The Dispatcher)
        // Synchronous FPGA block RAM fetch.  After a PC load, the requested
        // instruction is visible on the following cycle.
        current_instruction <= imem_array[pc[9:0]];
    end

    always_ff @(posedge clk) begin
        if (ext_imem_en && !ext_imem_write) begin
            ext_imem_rdata <= imem_array[ext_imem_addr];
        end
    end
`endif
`endif

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 32'd0;
        end else if (pc_load_en) begin
            pc <= pc_load_value;
        end else if (run_en) begin
            pc <= pc + 1;
        end
    end

    // bus control
    assign westbound_sel          = current_instruction[2:0];
    assign eastbound_sel          = current_instruction[5:3];
    assign westbound_consumer_sel = current_instruction[8:6];
    assign eastbound_consumer_sel = current_instruction[11:9];

    // 96-bit VLIW layout:
    //   [11:0]  bus control
    //   [28:12] mem0 read/write/address
    //   [45:29] mem1 read/write/address
    //   [62:46] MXM + fixed-point store controls
    //   [65:63] SXM transpose controls
    //   [78:66] VXM controls
    //   [79]    use the on-chip RoPE Q1.7 sine/cosine LUT
    //   [87:80] RoPE token position (0..255)
    //   [95:88] reserved

    // mem0 control
    assign mem0_read_en  = current_instruction[12];
    assign mem0_write_en = current_instruction[13];
    assign mem0_addr     = current_instruction[28:14];

    // mem1 control
    assign mem1_read_en  = current_instruction[29];
    assign mem1_write_en = current_instruction[30];
    assign mem1_addr     = current_instruction[45:31];

    // MXM control
    assign mxm_ingress_mode    = current_instruction[47:46];
    assign mxm_start           = current_instruction[48];
    assign mxm_clear           = current_instruction[49];
    assign mxm_e_row_sel       = current_instruction[52:50];
    assign mxm_e_col_sel       = current_instruction[55:53];
    assign mxm_e_valid_in      = current_instruction[56];
    assign mxm_input_is_signed = current_instruction[57];
    assign mxm_wght_is_signed  = current_instruction[58];
    assign mem_store_fmt       = current_instruction[62:61];

    // SXM transpose-only control
    assign sxm_opcode_input  = current_instruction[63] ? 12'h5A5 :
                               current_instruction[64] ? 12'hA5A :
                                                         12'h000;
    assign sxm_opcode_weight = 12'h000;
    assign sxm_load_from_west = current_instruction[65];
    
    // VXM control
    assign vxm_ctrl        = current_instruction[69:66];
    assign vxm_data_sel    = current_instruction[70];
    assign vxm_operand_sel = current_instruction[73:71];
    assign vxm_rmsnorm_en  = current_instruction[74];
    assign vxm_rope_en     = current_instruction[75];
    assign vxm_residual_op = current_instruction[78:76];
    assign vxm_rope_lut_en = current_instruction[79];
    assign vxm_rope_position = current_instruction[87:80];

endmodule
