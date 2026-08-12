`timescale 1ns/1ps
`include "lpu_pkg.sv"

module quant_q8_8_tb;
    localparam int LANES = 8;
    localparam int LANE_W = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid = 1'b0;
    quant_mode_e quant_mode_i = QUANT_SIGNED_INT8;
    logic signed [LANES*LANE_W-1:0] x_input = '0;
    logic out_valid;
    logic signed [LANES*8-1:0] q_row_out;
    logic [31:0] q_scale_out;

    always #5 clk = ~clk;

    quant #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .quant_mode_i(quant_mode_i),
        .x_input(x_input),
        .out_valid(out_valid),
        .q_row_out(q_row_out),
        .q_scale_out(q_scale_out)
    );

    task automatic set_lane(input int lane, input int value);
        x_input[lane*LANE_W +: LANE_W] = value[LANE_W-1:0];
    endtask

    function automatic int signed get_q_lane(input int lane);
        get_q_lane = $signed(q_row_out[lane*8 +: 8]);
    endfunction

    initial begin
        repeat (2) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        // Input is Q8.8: 1.0, -2.0, 0.5, -0.25.
        // Quant should preserve real values with output scale 2^-5:
        //  32 * 2^-5 = 1.0
        // -64 * 2^-5 = -2.0
        //  16 * 2^-5 = 0.5
        //  -8 * 2^-5 = -0.25
        x_input = '0;
        set_lane(0,  32'sd256);
        set_lane(1, -32'sd512);
        set_lane(2,  32'sd128);
        set_lane(3, -32'sd64);

        in_valid <= 1'b1;
        @(posedge clk);
        in_valid <= 1'b0;

        wait (out_valid);
        #1;

        if ($signed(q_scale_out[7:0]) != -8'sd5)
            $fatal(1, "expected output scale -5, got %0d", $signed(q_scale_out[7:0]));
        if (get_q_lane(0) != 32)
            $fatal(1, "lane0 expected 32, got %0d", get_q_lane(0));
        if (get_q_lane(1) != -64)
            $fatal(1, "lane1 expected -64, got %0d", get_q_lane(1));
        if (get_q_lane(2) != 16)
            $fatal(1, "lane2 expected 16, got %0d", get_q_lane(2));
        if (get_q_lane(3) != -8)
            $fatal(1, "lane3 expected -8, got %0d", get_q_lane(3));

        $display("QUANT_Q8_8_TEST_PASS scale=%0d lane0=%0d lane1=%0d lane2=%0d lane3=%0d",
                 $signed(q_scale_out[7:0]),
                 get_q_lane(0),
                 get_q_lane(1),
                 get_q_lane(2),
                 get_q_lane(3));
        $finish;
    end
endmodule
