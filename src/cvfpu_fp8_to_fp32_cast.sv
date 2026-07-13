`timescale 1ns/1ps

`include "cvfpu_compat.svh"

// Wrapper around CVFPU fpnew_top for scalar FP8 -> FP32 conversion.
//
// Notes:
// - This assumes CVFPU's built-in FP8 format (5 exponent / 2 mantissa).
// - fpnew_top expects narrower scalar operands to be NaN-boxed when the core
//   width is 32 and src_fmt_i selects FP8. That means bits [31:8] must be '1.
// - When HAVE_CVFPU is not defined, this module remains a compile-safe stub.
module cvfpu_fp8_to_fp32_cast (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,
    input  logic [7:0]  fp8_bits_i,
    output logic [31:0] result_o,
    output logic        done_o,
    output logic        busy_o
);

`ifdef HAVE_CVFPU
    localparam fpnew_pkg::fpu_features_t FPU_FEATURES = fpnew_pkg::RV32F_Xsflt;
    localparam fpnew_pkg::fpu_implementation_t FPU_IMPL = fpnew_pkg::DEFAULT_NOREGS;

    logic [2:0][31:0] operands_i;
    logic             in_ready_o;
    logic             out_valid_o;
    logic             cvfpu_busy_o;
    logic [31:0]      fpnew_result_o;
    logic             done_q;

    function automatic logic [31:0] nanbox_fp8(input logic [7:0] fp8_bits);
        return {24'hFF_FFFF, fp8_bits};
    endfunction

    assign operands_i[0] = nanbox_fp8(fp8_bits_i);
    assign operands_i[1] = '0;
    assign operands_i[2] = '0;

    fpnew_top #(
        .Features       (FPU_FEATURES),
        .Implementation (FPU_IMPL),
        .TagType        (logic)
    ) i_fp8_to_fp32 (
        .clk_i,
        .rst_ni,
        .operands_i,
        .rnd_mode_i     (fpnew_pkg::RNE),
        .op_i           (fpnew_pkg::F2F),
        .op_mod_i       (1'b0),
        .src_fmt_i      (fpnew_pkg::FP8),
        .dst_fmt_i      (fpnew_pkg::FP32),
        .int_fmt_i      (fpnew_pkg::INT32),
        .vectorial_op_i (1'b0),
        .tag_i          ('0),
        .simd_mask_i    ('1),
        .in_valid_i     (start_i),
        .in_ready_o     (in_ready_o),
        .flush_i        (1'b0),
        .result_o       (fpnew_result_o),
        .status_o       (/* unused */),
        .tag_o          (/* unused */),
        .out_valid_o    (out_valid_o),
        .out_ready_i    (1'b1),
        .busy_o         (cvfpu_busy_o),
        .early_valid_o  (/* unused */)
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

    function automatic logic [31:0] fp8_e5m2_to_fp32_bits(
        input logic [7:0] fp8_bits
    );
        logic       sign_bit;
        logic [4:0] exp_bits;
        logic [1:0] frac_bits;
        integer     exp_unbiased;
        real        value;
        real        scale;
        begin
            sign_bit = fp8_bits[7];
            exp_bits = fp8_bits[6:2];
            frac_bits = fp8_bits[1:0];

            if ((exp_bits == 5'd0) && (frac_bits == 2'd0)) begin
                fp8_e5m2_to_fp32_bits = sign_bit ? 32'h8000_0000 : 32'h0000_0000;
            end else if (exp_bits == 5'h1f) begin
                if (frac_bits == 2'd0)
                    fp8_e5m2_to_fp32_bits = sign_bit ? 32'hff80_0000 : 32'h7f80_0000;
                else
                    fp8_e5m2_to_fp32_bits = 32'h7fc0_0000;
            end else begin
                if (exp_bits == 5'd0) begin
                    scale = 2.0 ** (-16.0);
                    value = frac_bits * scale;
                end else begin
                    exp_unbiased = exp_bits - 15;
                    scale = 2.0 ** exp_unbiased;
                    value = (1.0 + (frac_bits / 4.0)) * scale;
                end

                if (sign_bit)
                    value = -value;

                fp8_e5m2_to_fp32_bits = f64_to_f32_bits($realtobits(value));
            end
        end
    endfunction

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            start_q  <= 1'b0;
            result_o <= 32'h0000_0000;
        end else begin
            start_q  <= start_i;
            if (start_i)
                result_o <= fp8_e5m2_to_fp32_bits(fp8_bits_i);
        end
    end

    assign done_o = start_q;
    assign busy_o = start_i | start_q;
`endif

endmodule
