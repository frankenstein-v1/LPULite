`timescale 1ns/1ns

module quant #(
    parameter int MULTIPLIER = 2032,
    parameter int SHIFT  = 16,
    parameter int LANES  = 4,
    parameter int LANE_W = 32
) (
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            in_valid,
    input  logic                            mode_softmax,
    input  logic signed [LANES*LANE_W-1:0]  x_input,

    output logic                            out_valid,
    output logic signed [LANES*8-1:0]       q_row_out
);

    localparam int OUT_W = LANES * 8;

    logic signed [LANES*LANE_W-1:0] ingress_data_reg;
    logic                           ingress_mode_reg;
    logic                           ingress_valid_reg;

    logic signed [LANE_W-1:0] lane_in [0:LANES-1];

    logic signed [7:0] regular_lane_q [0:LANES-1];
    logic        [7:0] softmax_lane_q [0:LANES-1];

    logic signed [OUT_W-1:0] regular_word;
    logic        [OUT_W-1:0] softmax_word;
    logic signed [OUT_W-1:0] mux_word;

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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ingress_data_reg  <= '0;
            ingress_mode_reg  <= 1'b0;
            ingress_valid_reg <= 1'b0;
            q_row_out         <= '0;
            out_valid         <= 1'b0;
        end else begin
            ingress_data_reg  <= x_input;
            ingress_mode_reg  <= mode_softmax;
            ingress_valid_reg <= in_valid;

            q_row_out <= mux_word;
            out_valid <= ingress_valid_reg;
        end
    end

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            lane_in[i] = ingress_data_reg[i*LANE_W +: LANE_W];
        end
    end

    //regular quant
    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            logic signed [LANE_W+31:0] product;
            logic signed [LANE_W+31:0] rounded;
            logic signed [LANE_W-1:0]  scaled_val;

            product = lane_in[i] * MULTIPLIER;

            if (product >= 0)
                rounded = product + ({{(LANE_W+31){1'b0}}, 1'b1} <<< (SHIFT - 1));
            else
                rounded = product - ({{(LANE_W+31){1'b0}}, 1'b1} <<< (SHIFT - 1));

            scaled_val = rounded >>> SHIFT;
            regular_lane_q[i] = clip_signed_q8(scaled_val);
        end
    end

    //softmax quant
    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            if (lane_in[i] <=0)
                softmax_lane_q[i] = 8'd0;
            else if (lane_in[i] >= 32'sd255)
                softmax_lane_q[i] = 8'd255;
            else 
                softmax_lane_q[i] = lane_in[i][7:0];
        end
    end

    always_comb begin
        regular_word = '0;
        softmax_word = '0;

        for (int i = 0; i < LANES; i++) begin
            regular_word[i*8 +: 8] = regular_lane_q[i];
            softmax_word[i*8 +: 8] = softmax_lane_q[i];
        end

        mux_word = ingress_mode_reg ? softmax_word : regular_word;
    end

endmodule
