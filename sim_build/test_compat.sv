`include "src/cvfpu_compat.svh"
module test_compat;
    logic [31:0] a_i, b_i;
    logic [31:0] result_o;
    logic sub_i;
    initial begin
        a_i = 32'h3f80_0000; // 1.0
        b_i = 32'h4000_0000; // 2.0
        sub_i = 1'b0;
        result_o = f64_to_f32_bits(
            $realtobits(
                sub_i
                    ? ($bitstoreal(f32_to_f64_bits(a_i)) - $bitstoreal(f32_to_f64_bits(b_i)))
                    : ($bitstoreal(f32_to_f64_bits(a_i)) + $bitstoreal(f32_to_f64_bits(b_i)))
            )
        );
        $display("result_o = %h", result_o);
        $finish;
    end
endmodule
