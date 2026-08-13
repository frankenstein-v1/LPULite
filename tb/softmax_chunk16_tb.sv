`timescale 1ns/1ps

module softmax_chunk16_tb;
    localparam int LANES = 8;
    localparam int LANE_W = 32;
    localparam int CHUNKS = 16;

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

    softmax #(.LANES(LANES), .LANE_W(LANE_W), .MAX_CHUNKS(CHUNKS)) dut (
        .clk, .rst_n, .in_valid, .x_in, .x_scale_i(-8'sd8), .in_ready,
        .out_valid, .y_out, .y_scale_o, .out_ready, .busy_o
    );

    initial begin
        repeat (2) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        // 16 rows x 8 lanes = 128 equal logits. Q0.7 encodes 1/128 as 1.
        in_valid <= 1'b1;
        repeat (CHUNKS) begin
            @(posedge clk);
            assert (in_ready) else $fatal(1, "softmax rejected an input chunk");
        end
        in_valid <= 1'b0;

        fork : wait_for_all_results
            begin
                wait (out_valid);
                for (int chunk = 0; chunk < CHUNKS; chunk++) begin
                    #1;
                    assert (out_valid) else $fatal(1, "softmax output stopped at chunk %0d", chunk);
                    assert (y_scale_o == -8'sd7) else $fatal(1, "bad probability scale");
                    for (int lane = 0; lane < LANES; lane++) begin
                        assert (y_out[lane*LANE_W +: LANE_W] == 32'd1)
                            else $fatal(1, "chunk %0d lane %0d probability=%0d expected=1",
                                        chunk, lane, y_out[lane*LANE_W +: LANE_W]);
                    end
                    @(posedge clk);
                end
                disable wait_for_all_results;
            end
            begin
                repeat (120) @(posedge clk);
                $fatal(1, "timed out waiting for 16-chunk softmax");
            end
        join

        $display("SOFTMAX_CHUNK16_TEST_PASS");
        $finish;
    end
endmodule
