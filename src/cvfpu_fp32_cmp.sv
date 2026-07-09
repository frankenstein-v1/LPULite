`timescale 1ns/1ps

// Wrapper around CVFPU fpnew_top for scalar FP32 comparisons.
//
// Compare mode encoding:
//   2'b00 -> a <= b
//   2'b01 -> a <  b
//   2'b10 -> a == b
//
// invert_i flips the boolean result, allowing !=, >, and >= forms.
module cvfpu_fp32_cmp (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,
    input  logic [1:0]  cmp_mode_i,
    input  logic        invert_i,
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    output logic        result_o,
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
    fpnew_pkg::roundmode_e cmp_rnd_mode;

    always_comb begin
        unique case (cmp_mode_i)
            2'b00:   cmp_rnd_mode = fpnew_pkg::RNE; // <=
            2'b01:   cmp_rnd_mode = fpnew_pkg::RTZ; // <
            2'b10:   cmp_rnd_mode = fpnew_pkg::RDN; // ==
            default: cmp_rnd_mode = fpnew_pkg::RNE;
        endcase
    end

    assign operands_i[0] = a_i;
    assign operands_i[1] = b_i;
    assign operands_i[2] = '0;

    fpnew_top #(
        .Features       (FPU_FEATURES),
        .Implementation (FPU_IMPL),
        .TagType        (logic)
    ) i_fp32_cmp (
        .clk_i,
        .rst_ni,
        .operands_i,
        .rnd_mode_i     (cmp_rnd_mode),
        .op_i           (fpnew_pkg::CMP),
        .op_mod_i       (invert_i),
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
            result_o <= 1'b0;
        end else begin
            done_q <= out_valid_o;
            if (out_valid_o)
                result_o <= fpnew_result_o[0];
        end
    end

    assign done_o = done_q;
    assign busy_o = cvfpu_busy_o | (start_i & ~in_ready_o) | out_valid_o | done_q;

`else
    logic start_q;
    shortreal a_value;
    shortreal b_value;
    logic cmp_result;

    always_comb begin
        a_value = $bitstoshortreal(a_i);
        b_value = $bitstoshortreal(b_i);
        unique case (cmp_mode_i)
            2'b00:   cmp_result = (a_value <= b_value);
            2'b01:   cmp_result = (a_value <  b_value);
            2'b10:   cmp_result = (a_value == b_value);
            default: cmp_result = 1'b0;
        endcase
        cmp_result = cmp_result ^ invert_i;
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            start_q  <= 1'b0;
            result_o <= 1'b0;
        end else begin
            start_q <= start_i;
            if (start_i)
                result_o <= cmp_result;
        end
    end

    assign done_o = start_q;
    assign busy_o = start_i | start_q;
`endif

endmodule
