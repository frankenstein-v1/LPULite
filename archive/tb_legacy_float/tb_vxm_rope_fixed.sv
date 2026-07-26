`timescale 1ns/1ps

module tb_vxm_rope_fixed;
    logic        clk;
    logic        rst_n;
    logic        start_i;
    logic [255:0] x_in;
    logic [63:0]  cos_q1_7;
    logic [63:0]  sin_q1_7;
    logic [255:0] y_out;
    logic        done_o;
    logic        busy_o;

    vxm_rope #(
        .LANES(8),
        .LANE_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(start_i),
        .x_in(x_in),
        .cos_q1_7(cos_q1_7),
        .sin_q1_7(sin_q1_7),
        .y_out(y_out),
        .done_o(done_o),
        .busy_o(busy_o)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        start_i = 0;
        x_in = '0;
        cos_q1_7 = '0;
        sin_q1_7 = '0;

        #20 rst_n = 1;
        #10;

        // Test vector inputs (32-bit signed ints for 8 lanes):
        x_in[ 0*32 +: 32] = 32'sd100;
        x_in[ 1*32 +: 32] = 32'sd50;
        x_in[ 2*32 +: 32] = -32'sd80;
        x_in[ 3*32 +: 32] = 32'sd120;
        x_in[ 4*32 +: 32] = 32'sd200;
        x_in[ 5*32 +: 32] = -32'sd100;
        x_in[ 6*32 +: 32] = 32'sd0;
        x_in[ 7*32 +: 32] = 32'sd64;

        // Cos/Sin Q1.7 trig values (scaled by 128):
        cos_q1_7[0*8 +: 8] = 8'sd123; cos_q1_7[1*8 +: 8] = 8'sd123;
        sin_q1_7[0*8 +: 8] = 8'sd37;  sin_q1_7[1*8 +: 8] = 8'sd37;

        cos_q1_7[2*8 +: 8] = 8'sd100; cos_q1_7[3*8 +: 8] = 8'sd100;
        sin_q1_7[2*8 +: 8] = -8'sd81; sin_q1_7[3*8 +: 8] = -8'sd81;

        cos_q1_7[4*8 +: 8] = 8'sd127; cos_q1_7[5*8 +: 8] = 8'sd127;
        sin_q1_7[4*8 +: 8] = 8'sd0;   sin_q1_7[5*8 +: 8] = 8'sd0;

        cos_q1_7[6*8 +: 8] = 8'sd0;   cos_q1_7[7*8 +: 8] = 8'sd0;
        sin_q1_7[6*8 +: 8] = 8'sd127; sin_q1_7[7*8 +: 8] = 8'sd127;

        start_i = 1;
        @(posedge clk);
        #1;
        start_i = 0;

        $display("y_out[0] = %d (expected %d)", $signed(y_out[0*32 +: 32]), (100*123 - 50*37) >>> 7);
        $display("y_out[1] = %d (expected %d)", $signed(y_out[1*32 +: 32]), (100*37 + 50*123) >>> 7);
        $display("y_out[2] = %d (expected %d)", $signed(y_out[2*32 +: 32]), ((-80)*100 - 120*(-81)) >>> 7);
        $display("y_out[3] = %d (expected %d)", $signed(y_out[3*32 +: 32]), ((-80)*(-81) + 120*100) >>> 7);

        if (done_o && $signed(y_out[0*32 +: 32]) == ((100*123 - 50*37) >>> 7)) begin
            $display("SUCCESS: Fixed-point RoPE test passed!");
        end else begin
            $display("FAILURE: Fixed-point RoPE test failed (done_o=%b).", done_o);
            $finish(1);
        end


        $finish;
    end
endmodule
