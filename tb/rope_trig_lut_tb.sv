`timescale 1ns/1ps

module rope_trig_lut_tb;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start_i = 1'b0;
    logic [7:0] position;
    logic [31:0] cos_pairs;
    logic [31:0] sin_pairs;
    logic [63:0] cos_lanes;
    logic [63:0] sin_lanes;
    logic [255:0] x_in;
    logic [255:0] y_out;
    logic done_o;
    logic busy_o;

    always #5 clk = ~clk;

    rope_trig_lut lut (
        .position_i(position),
        .cos_pairs_q1_7_o(cos_pairs),
        .sin_pairs_q1_7_o(sin_pairs)
    );

    genvar pair;
    generate
        for (pair = 0; pair < 4; pair++) begin : g_expand
            assign cos_lanes[(2*pair)*8 +: 8] = cos_pairs[pair*8 +: 8];
            assign cos_lanes[(2*pair+1)*8 +: 8] = cos_pairs[pair*8 +: 8];
            assign sin_lanes[(2*pair)*8 +: 8] = sin_pairs[pair*8 +: 8];
            assign sin_lanes[(2*pair+1)*8 +: 8] = sin_pairs[pair*8 +: 8];
        end
    endgenerate

    vxm_rope #(.LANES(8), .LANE_W(32)) dut (
        .clk(clk), .rst_n(rst_n), .start_i(start_i), .x_in(x_in),
        .cos_q1_7(cos_lanes), .sin_q1_7(sin_lanes),
        .y_out(y_out), .done_o(done_o), .busy_o(busy_o)
    );

    task automatic put_lane(input integer lane, input integer value);
        x_in[lane*32 +: 32] = value;
    endtask

    task automatic expect_lane(input integer lane, input integer expected);
        integer observed;
        begin
            observed = $signed(y_out[lane*32 +: 32]);
            if (observed !== expected) begin
                $display("FAIL lane %0d: expected %0d, got %0d", lane, expected, observed);
                $fatal(1);
            end
        end
    endtask

    initial begin
        position = 8'd0;
        x_in = '0;
        #1;
        if (cos_pairs !== 32'h7f7f7f7f || sin_pairs !== 32'h00000000)
            $fatal(1, "position zero must be the identity rotation");

        position = 8'd1;
        #1;
        if (cos_pairs !== 32'h7f7f7e45 || sin_pairs !== 32'h00010d6b)
            $fatal(1, "position-one LUT coefficients are incorrect");

        put_lane(0, 64);  put_lane(1, -32);
        put_lane(2, 96);  put_lane(3, 16);
        put_lane(4, -80); put_lane(5, 48);
        put_lane(6, 24);  put_lane(7, -56);

        #8 rst_n = 1'b1;
        #2 start_i = 1'b1;
        #10 start_i = 1'b0;
        #1;

        if (!done_o) $fatal(1, "RoPE result did not become valid");
        expect_lane(0, 61);  expect_lane(1, 36);
        expect_lane(2, 92);  expect_lane(3, 25);
        expect_lane(4, -80); expect_lane(5, 47);
        expect_lane(6, 23);  expect_lane(7, -56);
        $display("PASS: RoPE LUT and pairwise rotation");
        $finish;
    end
endmodule
