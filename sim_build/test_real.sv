module test_real;
    logic [63:0] bits;
    real r;
    always_comb begin
        bits = 64'hx;
        r = $bitstoreal(bits);
        $display("always_comb run");
    end
    initial begin
        #10;
        $finish;
    end
endmodule
