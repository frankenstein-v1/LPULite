`timescale 1ns/1ps

module rmsnorm_chunk_tb;
    localparam int LANES = 8;
    localparam int LANE_W = 32;
    localparam int CHUNKS = 2;
    localparam int ROW_W = LANES * LANE_W;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start_i = 1'b0;
    logic [ROW_W-1:0] x_in = '0;
    logic [ROW_W-1:0] gamma = '0;
    logic [ROW_W-1:0] beta = '0;
    logic in_ready;
    logic [ROW_W-1:0] y_out;
    logic done_o;
    logic out_ready = 1'b0;
    logic busy_o;

    always #5 clk = ~clk;

    rmsnorm #(
        .LANES(LANES),
        .LANE_W(LANE_W),
        .CHUNKS(CHUNKS)
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

    task automatic set_row(input int value);
        for (int lane = 0; lane < LANES; lane++) begin
            x_in[lane*LANE_W +: LANE_W] = value[LANE_W-1:0];
        end
    endtask

    initial begin
        int row0_lane;
        int row1_lane;

        for (int lane = 0; lane < LANES; lane++) begin
            gamma[lane*LANE_W +: LANE_W] = 32'sd128;
        end

        repeat (2) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        set_row(1);
        start_i <= 1'b1;
        @(posedge clk);
        start_i <= 1'b0;
        if (!in_ready) begin
            $fatal(1, "rmsnorm was not ready for first chunk");
        end

        @(posedge clk);
        set_row(3);
        start_i <= 1'b1;
        @(posedge clk);
        start_i <= 1'b0;

        wait (done_o);
        #1;
        row0_lane = $signed(y_out[31:0]);
        out_ready = 1'b1;
        @(posedge clk);
        #1;
        out_ready = 1'b0;
        wait (done_o);
        #1;
        row1_lane = $signed(y_out[31:0]);
        out_ready = 1'b1;
        @(posedge clk);
        #1;
        out_ready = 1'b0;

        if (row0_lane < 40 || row0_lane > 80) begin
            $fatal(1, "unexpected chunk0 output: %0d", row0_lane);
        end
        if (row1_lane < 130 || row1_lane > 220) begin
            $fatal(1, "unexpected chunk1 output: %0d", row1_lane);
        end
        if (row0_lane == row1_lane) begin
            $fatal(1, "RMSNorm behaved row-locally instead of chunking");
        end

        $display("RMSNORM_CHUNK_TEST_PASS row0=%0d row1=%0d", row0_lane, row1_lane);
        $finish;
    end
endmodule
