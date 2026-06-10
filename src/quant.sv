`timescale 1ns/1ns

module quant #(
    parameter int LANES  = 4,
    parameter int LANE_W = 32
) (
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            in_valid,
    input  logic                            mode_softmax,
    input  logic signed [LANES*LANE_W-1:0]  x_input,

    output logic                            out_valid,
    output logic signed [LANES*8-1:0]       q_row_out,
    output logic [31:0]                     q_scale_out
);

    localparam int OUT_W = LANES * 8;

    logic signed [LANES*LANE_W-1:0] ingress_data_reg;
    logic                           ingress_mode_reg;
    logic                           ingress_valid_reg;

    logic signed [7:0] regular_lane_q [0:LANES-1];
    logic        [7:0] softmax_lane_q [0:LANES-1];
    logic [7:0]        regular_row_shift;

    logic signed [OUT_W-1:0] regular_word;
    logic        [OUT_W-1:0] softmax_word;
    logic signed [OUT_W-1:0] mux_word;
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ingress_data_reg  <= '0;
            ingress_mode_reg  <= 1'b0;
            ingress_valid_reg <= 1'b0;
            q_row_out         <= '0;
            q_scale_out       <= '0;
            out_valid         <= 1'b0;
        end else begin
            ingress_data_reg  <= x_input;
            ingress_mode_reg  <= mode_softmax;
            ingress_valid_reg <= in_valid;

            q_row_out <= mux_word;
            q_scale_out <= scale_word;
            out_valid <= ingress_valid_reg;
        end
    end

    always_comb begin
        logic [LANE_W-1:0] max_abs_value;
        logic [LANE_W-1:0] lane_abs_value;
        logic [LANE_W-1:0] shifted_max_abs_value;
        logic signed [LANE_W-1:0] lane_in_val;
        logic signed [LANE_W-1:0] shifted_lane;
        int row_shift_next;

        regular_word = '0;
        softmax_word = '0;
        mux_word = '0;
        scale_word = '0;
        max_abs_value = '0;
        row_shift_next = 0;

        for (int i = 0; i < LANES; i++) begin
            lane_in_val = ingress_data_reg[i*LANE_W +: LANE_W];
            if (lane_in_val < 0)
                lane_abs_value = -lane_in_val;
            else
                lane_abs_value = lane_in_val;
            if (lane_abs_value > max_abs_value)
                max_abs_value = lane_abs_value;
        end

        shifted_max_abs_value = max_abs_value;
        for (int shift_idx = 0; shift_idx < (LANE_W - 1); shift_idx++) begin
            if (shifted_max_abs_value > 32'd127) begin
                shifted_max_abs_value = shifted_max_abs_value >> 1;
                row_shift_next = row_shift_next + 1;
            end
        end

        regular_row_shift = row_shift_next[7:0];
        scale_word = ingress_mode_reg ? 32'd0 : {24'd0, regular_row_shift};

        for (int i = 0; i < LANES; i++) begin
            lane_in_val = ingress_data_reg[i*LANE_W +: LANE_W];
            shifted_lane = round_shift_signed(lane_in_val, regular_row_shift);
            regular_lane_q[i] = clip_signed_q8(shifted_lane);
            softmax_lane_q[i] = (lane_in_val <= 0) ? 8'd0 :
                                (lane_in_val >= 32'sd255) ? 8'd255 :
                                lane_in_val[7:0];
            regular_word[i*8 +: 8] = regular_lane_q[i];
            softmax_word[i*8 +: 8] = softmax_lane_q[i];
        end

        mux_word = ingress_mode_reg ? softmax_word : regular_word;
    end

endmodule
