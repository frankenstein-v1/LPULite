`timescale 1ns/1ns

module vxm_four_row_observe_tb;
    localparam int LANES = 8;
    localparam int LANE_W = 32;
    localparam int ROW_W = LANES * LANE_W;

    logic clk;
    logic rst_n;
    logic [ROW_W-1:0] stream_in_data;
    logic [ROW_W-1:0] stream_in_bias;
    logic             in_valid;
    logic             in_ready;
    logic [3:0]       vxm_ctrl;
    logic             fp_quant_mode;
    logic [LANES*8-1:0] stream_out;
    logic [31:0]      stream_out_scale;
    logic             out_valid;
    logic             out_ready;

    vxm #(
        .LANES(LANES),
        .LANE_W(LANE_W),
        .ALU_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .stream_in_data(stream_in_data),
        .stream_in_bias(stream_in_bias),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .vxm_ctrl(vxm_ctrl),
        .rope_en(1'b0),
        .rope_cos_fp8(32'h3c3c_3c3c),
        .rope_sin_fp8(32'h0000_0000),
        .residual_op(3'd0),
        .layernorm_bypass(1'b1),
        .layernorm_gamma({LANES{32'h00000001}}),
        .layernorm_beta({LANES{32'h00000000}}),
        .fp_quant_mode(fp_quant_mode),
        .stream_out(stream_out),
        .stream_out_scale(stream_out_scale),
        .out_valid(out_valid),
        .out_ready(out_ready)
    );

    always #5 clk = ~clk;

    task automatic send_row(input logic [ROW_W-1:0] data_row);
        begin
            while (!in_ready)
                @(posedge clk);

            stream_in_data <= data_row;
            stream_in_bias <= '0;
            vxm_ctrl       <= 4'b1100; // bypass bias/relu, enable scale+softmax
            in_valid       <= 1'b1;
            @(posedge clk);
            in_valid       <= 1'b0;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        stream_in_data = '0;
        stream_in_bias = '0;
        in_valid = 1'b0;
        vxm_ctrl = '0;
        fp_quant_mode = 1'b0;
        out_ready = 1'b1;

        $dumpfile("tb/vxm_four_row_observe.vcd");
        $dumpvars(0, vxm_four_row_observe_tb);

        repeat (2) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        fork
            begin
                send_row({32'd0, 32'd2,  32'd4,  32'd6,  32'd8,  32'd10, 32'd12, 32'd16});
                send_row({32'd0, 32'd3,  32'd5,  32'd7,  32'd10, 32'd12, 32'd15, 32'd20});
                send_row({32'd0, 32'd1,  32'd3,  32'd4,  32'd6,  32'd8,  32'd10, 32'd12});
                send_row({32'd0, 32'd4,  32'd7,  32'd10, 32'd14, 32'd18, 32'd22, 32'd28});
            end
            begin
                repeat (250) @(posedge clk);
                $finish;
            end
        join
    end
endmodule
