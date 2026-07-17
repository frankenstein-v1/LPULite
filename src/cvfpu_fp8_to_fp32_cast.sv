`timescale 1ns/1ps

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
        logic        sign;
        logic [4:0]  exp_f8;
        logic [1:0]  frac_f8;
        logic [7:0]  exp_f32;
        logic [22:0] frac_f32;
        begin
            sign    = fp8_bits[7];
            exp_f8  = fp8_bits[6:2];
            frac_f8 = fp8_bits[1:0];

            if (exp_f8 == 5'd0) begin
                if (frac_f8 == 2'd0) begin
                    exp_f32  = 8'd0;
                    frac_f32 = 23'd0;
                end else if (frac_f8 == 2'b01) begin
                    exp_f32  = 127 - 16;
                    frac_f32 = 23'd0;
                end else if (frac_f8 == 2'b10) begin
                    exp_f32  = 127 - 15;
                    frac_f32 = 23'd0;
                end else begin
                    exp_f32  = 127 - 15;
                    frac_f32 = {1'b1, 22'd0};
                end
            end else if (exp_f8 == 5'h1f) begin
                if (frac_f8 == 2'd0) begin
                    exp_f32  = 8'hFF;
                    frac_f32 = 23'd0;
                end else begin
                    exp_f32  = 8'hFF;
                    frac_f32 = {1'b1, 22'd0};
                end
            end else begin
                exp_f32  = exp_f8 - 5'd15 + 8'd127;
                frac_f32 = {frac_f8, 21'd0};
            end

            fp8_e5m2_to_fp32_bits = {sign, exp_f32, frac_f32};
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
