`timescale 1ns/1ps
`include "lpu_pkg.sv"

module mem_row_dequant (
    input  mem_row_t mem_row_i,
    output mxm_row_t fp32_row_o
);

    function automatic logic [31:0] fp8_e5m2_to_fp32_bits(
        input logic [7:0] fp8_bits
    );
        logic        sign;
        logic [4:0]  exp_f8;
        logic [1:0]  frac_f8;
        logic [7:0]  exp_f32;
        logic [22:0] frac_f32;
        begin
            sign    = fp8_bits[7];
            exp_f8  = fp8_bits[6:2];
            frac_f8 = fp8_bits[1:0];

            if (exp_f8 == 5'd0) begin
                if (frac_f8 == 2'd0) begin
                    exp_f32  = 8'd0;
                    frac_f32 = 23'd0;
                end else if (frac_f8 == 2'b01) begin
                    exp_f32  = 8'd111;
                    frac_f32 = 23'd0;
                end else if (frac_f8 == 2'b10) begin
                    exp_f32  = 8'd112;
                    frac_f32 = 23'd0;
                end else begin
                    exp_f32  = 8'd112;
                    frac_f32 = {1'b1, 22'd0};
                end
            end else if (exp_f8 == 5'h1f) begin
                exp_f32  = 8'hFF;
                frac_f32 = (frac_f8 == 2'd0) ? 23'd0 : {1'b1, 22'd0};
            end else begin
                exp_f32  = {3'd0, exp_f8} + 8'd112;
                frac_f32 = {frac_f8, 21'd0};
            end

            fp8_e5m2_to_fp32_bits = {sign, exp_f32, frac_f32};
        end
    endfunction

    function automatic logic [31:0] fp32_scale_pow2(
        input logic [31:0] fp32_bits,
        input logic signed [7:0] scale_exp
    );
        logic        sign;
        logic [7:0]  exp_f32;
        logic [22:0] frac_f32;
        logic signed [9:0] scaled_exp;
        begin
            sign     = fp32_bits[31];
            exp_f32  = fp32_bits[30:23];
            frac_f32 = fp32_bits[22:0];

            if (exp_f32 == 8'd0) begin
                fp32_scale_pow2 = {sign, 31'd0};
            end else if (exp_f32 == 8'hFF) begin
                fp32_scale_pow2 = fp32_bits;
            end else begin
                scaled_exp = $signed({2'b00, exp_f32}) + $signed({{2{scale_exp[7]}}, scale_exp});
                if (scaled_exp <= 10'sd0) begin
                    fp32_scale_pow2 = {sign, 31'd0};
                end else if (scaled_exp >= 10'sd255) begin
                    fp32_scale_pow2 = {sign, 8'hFF, 23'd0};
                end else begin
                    fp32_scale_pow2 = {sign, scaled_exp[7:0], frac_f32};
                end
            end
        end
    endfunction

    generate
        genvar lane;
        for (lane = 0; lane < MXM_SIZE; lane++) begin : g_dequant_lane
            logic [31:0] lane_fp32;

            always @* begin
                lane_fp32 = fp8_e5m2_to_fp32_bits(mem_row_i[lane*8 +: 8]);
                fp32_row_o[lane*32 +: 32] = fp32_scale_pow2(
                    lane_fp32,
                    $signed(mem_row_i[71:64])
                );
            end
        end
    endgenerate

endmodule
