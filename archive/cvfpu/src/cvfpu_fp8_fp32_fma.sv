`timescale 1ns/1ps

// Wrapper around CVFPU fpnew_top for scalar widening fused multiply-add:
//   result = (fp8_a * fp8_b) + fp32_c
//
// Operands A/B are provided as raw FP8 bit patterns and NaN-boxed to the
// fpnew width. Operand C and the result are FP32 bit patterns.
module cvfpu_fp8_fp32_fma (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,
    input  logic [7:0]  fp8_a_i,
    input  logic [7:0]  fp8_b_i,
    input  logic [31:0] fp32_c_i,
    output logic [31:0] result_o,
    output logic        done_o,
    output logic        busy_o
);

`ifdef HAVE_CVFPU
    localparam fpnew_pkg::fpu_features_t FPU_FEATURES = fpnew_pkg::RV32F_Xsflt;
    localparam fpnew_pkg::fmt_logic_t FP_FORMATS = FPU_FEATURES.FpFmtMask;

    logic [2:0][31:0] operands_i;
    logic [fpnew_pkg::NUM_FP_FORMATS-1:0][2:0] is_boxed_i;
    logic                                       in_ready_o;
    logic                                       out_valid_o;
    logic                                       cvfpu_busy_o;
    logic [31:0]                                fpnew_result_o;
    logic                                       done_q;

    function automatic logic [31:0] nanbox_fp8(input logic [7:0] fp8_bits);
        return {24'hFF_FFFF, fp8_bits};
    endfunction

    assign operands_i[0] = nanbox_fp8(fp8_a_i);
    assign operands_i[1] = nanbox_fp8(fp8_b_i);
    assign operands_i[2] = fp32_c_i;
    assign is_boxed_i    = '1;

    fpnew_fma_multi #(
        .FpFmtConfig (FP_FORMATS),
        .NumPipeRegs (0),
        .PipeConfig  (fpnew_pkg::BEFORE),
        .TagType     (logic),
        .AuxType     (logic)
    ) i_fp8_fp32_fma (
        .clk_i,
        .rst_ni,
        .operands_i,
        .is_boxed_i      (is_boxed_i),
        .rnd_mode_i     (fpnew_pkg::RNE),
        .op_i           (fpnew_pkg::FMADD),
        .op_mod_i       (1'b0),
        .src_fmt_i      (fpnew_pkg::FP8),
        .src2_fmt_i     (fpnew_pkg::FP32),
        .dst_fmt_i      (fpnew_pkg::FP32),
        .tag_i          ('0),
        .mask_i         (1'b1),
        .aux_i          ('0),
        .in_valid_i     (start_i),
        .in_ready_o     (in_ready_o),
        .flush_i        (1'b0),
        .result_o       (fpnew_result_o),
        .status_o       (/* unused */),
        .extension_bit_o(/* unused */),
        .tag_o          (/* unused */),
        .mask_o         (/* unused */),
        .aux_o          (/* unused */),
        .out_valid_o    (out_valid_o),
        .out_ready_i    (1'b1),
        .busy_o         (cvfpu_busy_o),
        .reg_ena_i      ('1),
        .early_out_valid_o(/* unused */)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            done_q   <= 1'b0;
            result_o <= 32'h0000_0000;
        end else begin
            done_q <= out_valid_o;
            if (out_valid_o)
                result_o <= fpnew_result_o;
        end
    end

    assign done_o = done_q;
    assign busy_o = cvfpu_busy_o | (start_i & ~in_ready_o) | out_valid_o | done_q;

`else
    logic start_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            start_q  <= 1'b0;
            result_o <= 32'h0000_0000;
        end else begin
            start_q  <= start_i;
            result_o <= 32'h0000_0000;
        end
    end

    assign done_o = start_q;
    assign busy_o = start_i | start_q;
`endif

endmodule
