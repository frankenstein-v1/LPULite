`timescale 1ns/1ps

module mac_mixed_scale_tb;
    logic clk = 1'b0;
    logic rst = 1'b1;
    logic clear = 1'b0;
    logic en = 1'b0;
    logic signed [8:0] input_i = '0;
    logic signed [8:0] weight_i = '0;
    logic signed [7:0] input_scale_i = '0;
    logic signed [7:0] weight_scale_i = '0;
    logic signed [31:0] acc_o;
    logic signed [7:0] acc_scale_o;
    logic signed [17:0] product_o;

    always #5 clk = ~clk;

    mac #(
        .INPUT_W(9),
        .WEIGHT_W(9),
        .PRODUCT_W(18),
        .ACC_W(32),
        .SCALE_W(8)
    ) dut (
        .clk(clk),
        .rst(rst),
        .clear(clear),
        .en(en),
        .input_i(input_i),
        .weight_i(weight_i),
        .input_scale_i(input_scale_i),
        .weight_scale_i(weight_scale_i),
        .acc_valid_i(1'b0),
        .aligned_scale_i('0),
        .acc_shift_i('0),
        .product_shift_i('0),
        .acc_o(acc_o),
        .acc_scale_o(acc_scale_o),
        .product_o(product_o)
    );

    task automatic accumulate(
        input integer input_value,
        input integer weight_value,
        input integer input_scale,
        input integer weight_scale
    );
        begin
            input_i = input_value;
            weight_i = weight_value;
            input_scale_i = input_scale;
            weight_scale_i = weight_scale;
            en = 1'b1;
            @(posedge clk);
            #1;
            en = 1'b0;
        end
    endtask

    task automatic clear_accumulator;
        begin
            clear = 1'b1;
            @(posedge clk);
            #1;
            clear = 1'b0;
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        #1;
        rst = 1'b0;

        // 2*1*2^-4 + 3*1*2^-2 = 14*2^-4.
        accumulate(2, 1, -4, 0);
        accumulate(3, 1, -2, 0);
        if ((acc_o !== 32'sd14) || (acc_scale_o !== -8'sd4)) begin
            $fatal(1, "mixed-scale forward order failed: acc=%0d scale=%0d", acc_o, acc_scale_o);
        end

        // Accumulation must be independent of exponent arrival order.
        clear_accumulator();
        accumulate(3, 1, -2, 0);
        accumulate(2, 1, -4, 0);
        if ((acc_o !== 32'sd14) || (acc_scale_o !== -8'sd4)) begin
            $fatal(1, "mixed-scale reverse order failed: acc=%0d scale=%0d", acc_o, acc_scale_o);
        end

        // Same-scale behavior remains the ordinary raw dot-product sum.
        clear_accumulator();
        accumulate(-7, 5, -3, -2);
        accumulate(4, 6, -3, -2);
        if ((acc_o !== -32'sd11) || (acc_scale_o !== -8'sd5)) begin
            $fatal(1, "same-scale accumulation failed: acc=%0d scale=%0d", acc_o, acc_scale_o);
        end

        $display("MAC_MIXED_SCALE_TEST_PASS");
        $finish;
    end
endmodule
