`timescale 1ns/1ps

module tb_rmsnorm_fixed;
    logic        clk;
    logic        rst_n;
    logic        start_i;
    logic [255:0] x_in;
    logic [255:0] gamma;
    logic [255:0] beta;
    logic        in_ready;
    logic [255:0] y_out;
    logic        done_o;
    logic        out_ready;
    logic        busy_o;

    rmsnorm #(
        .LANES(8),
        .LANE_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(start_i),
        .x_in(x_in),
        .gamma(gamma),
        .beta(beta),
        .in_ready(in_ready),
        .y_out(y_out),
        .done_o(done_o),
        .out_ready(out_ready),
        .busy_o(busy_o)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        start_i = 0;
        x_in = '0;
        gamma = '0;
        beta = '0;
        out_ready = 1;

        #20 rst_n = 1;
        #10;

        // Set input vector x_in (32-bit signed ints for 8 lanes):
        x_in[ 0*32 +: 32] = 32'sd100;
        x_in[ 1*32 +: 32] = 32'sd50;
        x_in[ 2*32 +: 32] = -32'sd80;
        x_in[ 3*32 +: 32] = 32'sd120;
        x_in[ 4*32 +: 32] = 32'sd200;
        x_in[ 5*32 +: 32] = -32'sd100;
        x_in[ 6*32 +: 32] = 32'sd0;
        x_in[ 7*32 +: 32] = 32'sd64;

        // Gamma parameters = identity scaling (0 or 128)
        for (int i = 0; i < 8; i++) begin
            gamma[i*32 +: 32] = 32'sd128;
        end

        start_i = 1;
        @(posedge clk);
        #1;
        start_i = 0;

        $display("y_out[0] = %d", $signed(y_out[0*32 +: 32]));
        $display("y_out[1] = %d", $signed(y_out[1*32 +: 32]));
        $display("y_out[2] = %d", $signed(y_out[2*32 +: 32]));
        $display("y_out[3] = %d", $signed(y_out[3*32 +: 32]));
        $display("y_out[4] = %d", $signed(y_out[4*32 +: 32]));
        $display("y_out[5] = %d", $signed(y_out[5*32 +: 32]));

        if (done_o) begin
            $display("SUCCESS: Fixed-point RMSNorm test passed!");
        end else begin
            $display("FAILURE: Fixed-point RMSNorm test failed (done_o=%b).", done_o);
            $finish(1);
        end

        $finish;
    end
endmodule
