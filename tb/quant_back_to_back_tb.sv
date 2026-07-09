`timescale 1ns/1ps

module quant_back_to_back_tb;
    localparam int LANES  = 4;
    localparam int LANE_W = 32;

    logic                            clk;
    logic                            rst_n;
    logic                            in_valid;
    logic                            mode_softmax;
    logic                            fp_quant_mode;
    logic                            softmax_input_is_fp;
    logic signed [LANES*LANE_W-1:0]  x_input;
    logic                            out_valid;
    logic signed [LANES*8-1:0]       q_row_out;
    logic [31:0]                     q_scale_out;

    quant #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .mode_softmax(mode_softmax),
        .fp_quant_mode(fp_quant_mode),
        .softmax_input_is_fp(softmax_input_is_fp),
        .x_input(x_input),
        .out_valid(out_valid),
        .q_row_out(q_row_out),
        .q_scale_out(q_scale_out)
    );

    always #5 clk = ~clk;

    function automatic logic signed [LANES*LANE_W-1:0] pack_input_row(
        input logic signed [LANE_W-1:0] lane0,
        input logic signed [LANE_W-1:0] lane1,
        input logic signed [LANE_W-1:0] lane2,
        input logic signed [LANE_W-1:0] lane3
    );
        begin
            pack_input_row = {lane3, lane2, lane1, lane0};
        end
    endfunction

    initial begin
        $dumpfile("build/quant_back_to_back.vcd");
        $dumpvars(0, quant_back_to_back_tb);

        clk = 1'b0;
        rst_n = 1'b0;
        in_valid = 1'b0;
        mode_softmax = 1'b0;
        fp_quant_mode = 1'b0;
        softmax_input_is_fp = 1'b0;
        x_input = '0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Transaction 1: regular quant mode.
        in_valid = 1'b1;
        mode_softmax = 1'b0;
        fp_quant_mode = 1'b0;
        softmax_input_is_fp = 1'b0;
        x_input = pack_input_row(-32'sd4096, -32'sd2048, 32'sd2048, 32'sd4096);
        @(posedge clk);

        // Transaction 2: softmax quant mode immediately after.
        in_valid = 1'b1;
        mode_softmax = 1'b1;
        fp_quant_mode = 1'b0;
        softmax_input_is_fp = 1'b0;
        x_input = pack_input_row(32'sd4, 32'sd28, 32'sd96, 32'sd128);
        @(posedge clk);

        // Deassert ingress after the back-to-back pair.
        in_valid = 1'b0;
        mode_softmax = 1'b0;
        fp_quant_mode = 1'b0;
        softmax_input_is_fp = 1'b0;
        x_input = '0;

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule
