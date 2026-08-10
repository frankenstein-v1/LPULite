`timescale 1ns/1ps

module softmax_lut_tb;
    localparam int LANES = 8;
    localparam int LANE_W = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid = 1'b0;
    logic [LANES*LANE_W-1:0] x_in = '0;
    logic in_ready;
    logic out_valid;
    logic [LANES*LANE_W-1:0] y_out;
    logic signed [7:0] y_scale_o;
    logic out_ready = 1'b1;
    logic busy_o;

    always #5 clk = ~clk;

    softmax #(
        .LANES(LANES),
        .LANE_W(LANE_W),
        .MAX_CHUNKS(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .x_in(x_in),
        .x_scale_i(-8'sd8),
        .in_ready(in_ready),
        .out_valid(out_valid),
        .y_out(y_out),
        .y_scale_o(y_scale_o),
        .out_ready(out_ready),
        .busy_o(busy_o)
    );

    initial begin
        repeat (2) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        // Eight equal logits must normalize to eight copies of 1/8. With the
        // output scale 2^-7, each encoded probability is exactly 16.
        in_valid <= 1'b1;
        x_in <= '0;
        @(posedge clk);
        in_valid <= 1'b0;

        fork : wait_for_result
            begin
                wait (out_valid);
                #1;
                assert (y_scale_o == -8'sd7)
                    else $fatal(1, "unexpected probability scale %0d", y_scale_o);
                for (int lane = 0; lane < LANES; lane++) begin
                    assert (y_out[lane*LANE_W +: LANE_W] == 32'd16)
                        else $fatal(1, "lane %0d probability was %0d, expected 16",
                                    lane, y_out[lane*LANE_W +: LANE_W]);
                end
                disable wait_for_result;
            end
            begin
                repeat (20) @(posedge clk);
                $fatal(1, "timed out waiting for LUT-normalized softmax output");
            end
        join

        $display("PASS: reciprocal-LUT softmax produced eight equal Q0.7 probabilities");
        $finish;
    end
endmodule
