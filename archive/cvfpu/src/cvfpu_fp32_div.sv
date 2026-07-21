`timescale 1ns/1ps

`include "cvfpu_compat.svh"

// Wrapper around CVFPU fpnew_top for scalar FP32 divide:
//   result = dividend / divisor
module cvfpu_fp32_div (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,
    input  logic [31:0] dividend_i,
    input  logic [31:0] divisor_i,
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

    assign operands_i[0] = dividend_i;
    assign operands_i[1] = divisor_i;
    assign operands_i[2] = '0;

    fpnew_top #(
        .Features       (FPU_FEATURES),
        .Implementation (FPU_IMPL),
        .TagType        (logic)
    ) i_fp32_div (
        .clk_i,
        .rst_ni,
        .operands_i,
        .rnd_mode_i     (fpnew_pkg::RNE),
        .op_i           (fpnew_pkg::DIV),
        .op_mod_i       (1'b0),
        .src_fmt_i      (fpnew_pkg::FP32),
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

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            start_q  <= 1'b0;
            result_o <= 32'h0000_0000;
        end else begin
            start_q <= start_i;
            if (start_i) begin
                result_o <= f64_to_f32_bits(
                    $realtobits(
                        $bitstoreal(f32_to_f64_bits(dividend_i)) / $bitstoreal(f32_to_f64_bits(divisor_i))
                    )
                );
            end
        end
    end

    assign done_o = start_q;
    assign busy_o = start_i | start_q;
`endif

endmodule
