`timescale 1ns/1ps

// Minimal elaboration target for the CVFPU-backed wrapper path.
//
// This is not trying to verify floating-point correctness yet. Its only job is
// to prove that:
// 1. the vendored CVFPU tree is present,
// 2. the wrapper modules bind against fpnew_top,
// 3. the design elaborates with HAVE_CVFPU enabled.
module cvfpu_compile_smoke_top;
    logic        clk;
    logic        rst_n;
    logic        start_cast;
    logic        start_fma;
    logic [7:0]  fp8_in;
    logic [31:0] fp32_a, fp32_b, fp32_c;
    logic [31:0] cast_out;
    logic [31:0] fma_out;
    logic        cast_done;
    logic        cast_busy;
    logic        fma_done;
    logic        fma_busy;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        rst_n      = 1'b0;
        start_cast = 1'b0;
        start_fma  = 1'b0;
        fp8_in     = 8'h3C;
        fp32_a     = 32'h3F80_0000; // 1.0f
        fp32_b     = 32'h4000_0000; // 2.0f
        fp32_c     = 32'h4040_0000; // 3.0f

        #20;
        rst_n = 1'b1;

        #10;
        start_cast = 1'b1;
        start_fma  = 1'b1;

        #10;
        start_cast = 1'b0;
        start_fma  = 1'b0;

        #100;
        $finish;
    end

    cvfpu_fp8_to_fp32_cast u_cast (
        .clk_i      (clk),
        .rst_ni     (rst_n),
        .start_i    (start_cast),
        .fp8_bits_i (fp8_in),
        .result_o   (cast_out),
        .done_o     (cast_done),
        .busy_o     (cast_busy)
    );

    cvfpu_fp32_fma u_fma (
        .clk_i          (clk),
        .rst_ni         (rst_n),
        .start_i        (start_fma),
        .multiplicand_i (fp32_a),
        .multiplier_i   (fp32_b),
        .addend_i       (fp32_c),
        .result_o       (fma_out),
        .done_o         (fma_done),
        .busy_o         (fma_busy)
    );
endmodule
