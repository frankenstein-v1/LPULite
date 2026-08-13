`timescale 1ns/1ps
`include "lpu_pkg.sv"

module sxm_scaled_broadcast_tb;
    localparam int LANES = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [11:0] opcode_input = '0;
    logic [11:0] opcode_weight = '0;
    logic load_from_west = 1'b1;
    superlane_t eastbound_in = '0;
    logic eastbound_valid_i = 1'b0;
    logic signed [7:0] eastbound_scale_i = '0;
    superlane_t eastbound_out;
    superlane_t westbound_in = '0;
    logic westbound_valid_i = 1'b1;
    logic signed [7:0] westbound_scale_i = -8'sd5;
    superlane_t westbound_out;
    logic signed [7:0] emit_scale_o;
    logic emit_valid;

    always #5 clk = ~clk;

    sxm #(.LANES(LANES)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .opcode_input(opcode_input),
        .opcode_weight(opcode_weight),
        .load_from_west(load_from_west),
        .eastbound_in(eastbound_in),
        .eastbound_valid_i(eastbound_valid_i),
        .eastbound_scale_i(eastbound_scale_i),
        .eastbound_out(eastbound_out),
        .westbound_in(westbound_in),
        .westbound_valid_i(westbound_valid_i),
        .westbound_scale_i(westbound_scale_i),
        .westbound_out(westbound_out),
        .emit_scale_o(emit_scale_o),
        .emit_valid(emit_valid)
    );

    initial begin
        for (int lane = 0; lane < LANES; lane++) begin
            westbound_in[lane*8 +: 8] = 8'(lane + 1);
        end

        repeat (2) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        opcode_input <= 12'h5A5;
        @(posedge clk);
        opcode_input <= 12'h000;
        repeat (7) @(posedge clk);

        opcode_input <= 12'hA5A;
        @(posedge clk);
        opcode_input <= 12'h000;
        for (int row = 0; row < LANES; row++) begin
            #1;
            assert (emit_valid) else $fatal(1, "SXM row %0d was not valid", row);
            assert (emit_scale_o == -8'sd5)
                else $fatal(1, "SXM lost block scale: got %0d", emit_scale_o);
            for (int lane = 0; lane < LANES; lane++) begin
                assert (westbound_out[lane*8 +: 8] == 8'(row + 1))
                    else $fatal(1, "broadcast row %0d lane %0d = %0d, expected %0d",
                                row, lane, westbound_out[lane*8 +: 8], row + 1);
            end
            @(posedge clk);
        end

        $display("SXM_SCALED_BROADCAST_TEST_PASS");
        $finish;
    end
endmodule
