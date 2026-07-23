`timescale 1ns/1ns
`include "lpu_pkg.sv"

module quant #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32,
    parameter logic signed [7:0] SOFTMAX_PROB_SCALE = -8'sd7
) (
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            in_valid,
    input  quant_mode_e                     quant_mode_i,
    input  logic signed [LANES*LANE_W-1:0]  x_input,

    output logic                            out_valid,
    output logic signed [LANES*8-1:0]       q_row_out,
    output logic [31:0]                     q_scale_out
);

    localparam int OUT_W = LANES * 8;

    logic signed [LANES*LANE_W-1:0] ingress_data_reg;
    quant_mode_e                    ingress_mode_reg;
    logic                           ingress_valid_reg;

    logic signed [7:0] regular_lane_q [0:LANES-1];
    logic        [7:0] regular_fp8_q [0:LANES-1];
    logic        [7:0] softmax_lane_q [0:LANES-1];
    logic [7:0]        regular_row_shift;
    logic signed [7:0] regular_row_scale_exp;

    logic signed [OUT_W-1:0] regular_word;
    logic        [OUT_W-1:0] regular_fp8_word;
    logic        [OUT_W-1:0] softmax_word;
    logic        [OUT_W-1:0] mux_word;
    logic [31:0]             scale_word;

    function automatic logic signed [7:0] clip_signed_q8(
        input logic signed [LANE_W-1:0] value
    );
        begin
            if (value > 32'sd127)
                clip_signed_q8 = 8'sd127;
            else if (value < -32'sd127)
                clip_signed_q8 = -8'sd127;
            else
                clip_signed_q8 = value[7:0];
        end
    endfunction

    function automatic logic signed [LANE_W-1:0] round_shift_signed(
        input logic signed [LANE_W-1:0] value,
        input logic [7:0]               shift_amount
    );
        logic signed [LANE_W-1:0] rounding_step;
        logic signed [LANE_W-1:0] adjusted_value;
        begin
            if (shift_amount == 0) begin
                round_shift_signed = value;
            end else begin
                rounding_step = $signed({{(LANE_W-1){1'b0}}, 1'b1}) <<< (shift_amount - 1);
                adjusted_value = (value >= 0) ? (value + rounding_step) : (value - rounding_step);
                round_shift_signed = adjusted_value >>> shift_amount;
            end
        end
    endfunction

    function automatic logic [30:0] fp32_abs_mag(
        input logic [31:0] fp_bits
    );
        begin
            fp32_abs_mag = {1'b0, fp_bits[30:0]};
        end
    endfunction

    function automatic logic signed [7:0] fp32_floor_log2_abs(
        input logic [31:0] fp_bits
    );
        logic [7:0]  exp_bits;
        logic [22:0] frac_bits;
        begin
            exp_bits = fp_bits[30:23];
            frac_bits = fp_bits[22:0];

            if ((exp_bits == 8'h00) && (frac_bits == 23'd0))
                fp32_floor_log2_abs = 8'sd0;
            else if (exp_bits == 8'h00)
                fp32_floor_log2_abs = -8'sd126;
            else
                fp32_floor_log2_abs = $signed({1'b0, exp_bits}) - 8'sd127;
        end
    endfunction

    function automatic logic [31:0] fp32_scale_by_pow2(
        input logic [31:0]        fp_bits,
        input logic signed [7:0]  scale_exp
    );
        logic        sign_bit;
        logic [7:0]  exp_bits;
        logic [22:0] frac_bits;
        integer      adjusted_exp;
        begin
            sign_bit = fp_bits[31];
            exp_bits = fp_bits[30:23];
            frac_bits = fp_bits[22:0];

            if ((exp_bits == 8'h00) && (frac_bits == 23'd0)) begin
                fp32_scale_by_pow2 = fp_bits;
            end else if (exp_bits == 8'hff) begin
                fp32_scale_by_pow2 = fp_bits;
            end else if (exp_bits == 8'h00) begin
                // Keep subnormals simple in this quantizer: underflow them to signed zero
                fp32_scale_by_pow2 = {sign_bit, 31'd0};
            end else begin
                adjusted_exp = $signed({1'b0, exp_bits}) + scale_exp;
                if (adjusted_exp >= 255)
                    fp32_scale_by_pow2 = {sign_bit, 8'hff, 23'd0};
                else if (adjusted_exp <= 0)
                    fp32_scale_by_pow2 = {sign_bit, 31'd0};
                else
                    fp32_scale_by_pow2 = {sign_bit, adjusted_exp[7:0], frac_bits};
            end
        end
    endfunction

    function automatic logic [7:0] fp32_to_fp8_e5m2(
        input logic [31:0] fp_bits
    );
        logic        sign_bit;
        logic [7:0]  exp_bits;
        logic [22:0] frac_bits;
        logic [23:0] mantissa_full;
        logic [2:0]  mantissa_q;
        logic        guard_bit;
        logic        sticky_bit;
        integer      fp8_exp;
        begin
            sign_bit = fp_bits[31];
            exp_bits = fp_bits[30:23];
            frac_bits = fp_bits[22:0];

            if ((exp_bits == 8'h00) && (frac_bits == 23'd0)) begin
                fp32_to_fp8_e5m2 = {sign_bit, 7'd0};
            end else if (exp_bits == 8'hff) begin
                if (frac_bits != 23'd0)
                    fp32_to_fp8_e5m2 = {sign_bit, 5'h1f, 2'b01};
                else
                    fp32_to_fp8_e5m2 = {sign_bit, 5'h1f, 2'b00};
            end else if (exp_bits == 8'h00) begin
                fp32_to_fp8_e5m2 = {sign_bit, 7'd0};
            end else begin
                fp8_exp = exp_bits - 127 + 15;
                if (fp8_exp <= 0) begin
                    fp32_to_fp8_e5m2 = {sign_bit, 7'd0};
                end else begin
                    mantissa_full = {1'b1, frac_bits};
                    mantissa_q = mantissa_full[23:21];
                    guard_bit = mantissa_full[20];
                    sticky_bit = |mantissa_full[19:0];

                    if (guard_bit && (sticky_bit || mantissa_q[0]))
                        mantissa_q = mantissa_q + 3'd1;

                    if (mantissa_q == 3'd0) begin
                        mantissa_q = 3'd4;
                        fp8_exp = fp8_exp + 1;
                    end

                    if (fp8_exp >= 31) begin
                        fp32_to_fp8_e5m2 = {sign_bit, 5'h1f, 2'b00};
                    end else begin
                        fp32_to_fp8_e5m2 = {sign_bit, fp8_exp[4:0], mantissa_q[1:0]};
                    end
                end
            end
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ingress_data_reg  <= '0;
            ingress_mode_reg  <= QUANT_SIGNED_INT8;
            ingress_valid_reg <= 1'b0;
            q_row_out         <= '0;
            q_scale_out       <= '0;
            out_valid         <= 1'b0;
        end else begin
            ingress_data_reg  <= x_input;
            ingress_mode_reg  <= quant_mode_i;
            ingress_valid_reg <= in_valid;

            q_row_out <= mux_word;
            q_scale_out <= scale_word;
            out_valid <= ingress_valid_reg;
        end
    end

    always @* begin
        logic [LANE_W-1:0] max_abs_value;
        logic [LANE_W-1:0] lane_abs_value;
        logic [LANE_W-1:0] shifted_max_abs_value;
        logic signed [LANE_W-1:0] lane_in_val;
        logic signed [LANE_W-1:0] shifted_lane;
        logic [30:0]              max_abs_mag_fp;
        logic [30:0]              lane_abs_mag_fp;
        logic signed [7:0]        row_scale_exp_next;
        logic [31:0]              scaled_fp_lane_bits;
        int row_shift_next;

        regular_word = '0;
        regular_fp8_word = '0;
        softmax_word = '0;
        mux_word = '0;
        scale_word = '0;
        max_abs_value = '0;
        max_abs_mag_fp = 31'd0;
        row_shift_next = 0;
        row_scale_exp_next = 8'sd0;

        for (int i = 0; i < LANES; i++) begin
            lane_in_val = ingress_data_reg[i*LANE_W +: LANE_W];
            if (lane_in_val < 0)
                lane_abs_value = -lane_in_val;
            else
                lane_abs_value = lane_in_val;
            if (lane_abs_value > max_abs_value)
                max_abs_value = lane_abs_value;

            lane_abs_mag_fp = fp32_abs_mag(ingress_data_reg[i*LANE_W +: LANE_W]);
            if (lane_abs_mag_fp > max_abs_mag_fp)
                max_abs_mag_fp = lane_abs_mag_fp;
        end

        shifted_max_abs_value = max_abs_value;
        for (int shift_idx = 0; shift_idx < (LANE_W - 1); shift_idx++) begin
            if (shifted_max_abs_value > 32'd127) begin
                shifted_max_abs_value = shifted_max_abs_value >> 1;
                row_shift_next = row_shift_next + 1;
            end
        end

        regular_row_shift = row_shift_next[7:0];
        if (max_abs_mag_fp != 31'd0)
            row_scale_exp_next = fp32_floor_log2_abs({1'b0, max_abs_mag_fp});
        regular_row_scale_exp = row_scale_exp_next;

        for (int i = 0; i < LANES; i++) begin
            lane_in_val = ingress_data_reg[i*LANE_W +: LANE_W];
            shifted_lane = round_shift_signed(lane_in_val, regular_row_shift);
            regular_lane_q[i] = clip_signed_q8(shifted_lane);

            scaled_fp_lane_bits = fp32_scale_by_pow2(
                ingress_data_reg[i*LANE_W +: LANE_W],
                -regular_row_scale_exp
            );
            regular_fp8_q[i] = fp32_to_fp8_e5m2(scaled_fp_lane_bits);

            softmax_lane_q[i] = (lane_in_val <= 0) ? 8'd0 :
                                (lane_in_val >= 32'sd255) ? 8'd255 :
                                lane_in_val[7:0];
            regular_word[i*8 +: 8] = regular_lane_q[i];
            regular_fp8_word[i*8 +: 8] = regular_fp8_q[i];
            softmax_word[i*8 +: 8] = softmax_lane_q[i];
        end

        unique case (ingress_mode_reg)
            QUANT_SOFTMAX_U8: begin
                mux_word = softmax_word;
                scale_word = {{24{SOFTMAX_PROB_SCALE[7]}}, SOFTMAX_PROB_SCALE};
            end
            QUANT_FP8_E5M2: begin
                mux_word = regular_fp8_word;
                scale_word = {{24{regular_row_scale_exp[7]}}, regular_row_scale_exp[7:0]};
            end
            default: begin
                mux_word = regular_word;
                scale_word = {24'd0, regular_row_shift};
            end
        endcase
    end

endmodule
