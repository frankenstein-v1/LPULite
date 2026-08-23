`timescale 1ns/1ps

// Fixed-point VXM RoPE (Rotary Position Embedding) stage.
//
// Computes 2D vector rotations on adjacent element pairs using 32-bit signed
// fixed-point inputs (x_in) and 8-bit signed fixed-point trig values (cos_q1_7/sin_q1_7).
// Intermediate 40-bit products are combined and arithmetic right-shifted by 7 (>>> 7).
module vxm_rope #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*8-1:0]      cos_q1_7, // 8-lane signed int8 fixed-point cos (Q1.7)
    input  logic [LANES*8-1:0]      sin_q1_7, // 8-lane signed int8 fixed-point sin (Q1.7)
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    output logic                    busy_o
);

    logic [LANES*LANE_W-1:0] y_next;
    logic                    busy_q;

    genvar i;
    generate
        for (i = 0; i < LANES/2; i++) begin : g_rope_pairs
            localparam int EVEN = 2 * i;
            localparam int ODD  = 2 * i + 1;

            logic signed [LANE_W-1:0] x_even;
            logic signed [LANE_W-1:0] x_odd;
            logic signed [7:0]        c_val;
            logic signed [7:0]        s_val;

`ifdef LPULITE_VXM_LOGIC_MULT
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod0;
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod1;
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod2;
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod3;
`else
            logic signed [LANE_W+8-1:0] prod0;
            logic signed [LANE_W+8-1:0] prod1;
            logic signed [LANE_W+8-1:0] prod2;
            logic signed [LANE_W+8-1:0] prod3;
`endif

            logic signed [LANE_W+8-1:0] res_even_full;
            logic signed [LANE_W+8-1:0] res_odd_full;

            logic signed [LANE_W-1:0]   res_even;
            logic signed [LANE_W-1:0]   res_odd;

            assign x_even = $signed(x_in[EVEN*LANE_W +: LANE_W]);
            assign x_odd  = $signed(x_in[ODD*LANE_W  +: LANE_W]);
            assign c_val  = $signed(cos_q1_7[EVEN*8   +: 8]);
            assign s_val  = $signed(sin_q1_7[EVEN*8   +: 8]);

            assign prod0 = x_even * c_val;
            assign prod1 = x_odd  * s_val;
            assign prod2 = x_even * s_val;
            assign prod3 = x_odd  * c_val;

            assign res_even_full = prod0 - prod1;
            assign res_odd_full  = prod2 + prod3;

            // Arithmetic right shift by 7 for Q1.7 fixed-point scaling
            assign res_even = LANE_W'(res_even_full >>> 7);
            assign res_odd  = LANE_W'(res_odd_full  >>> 7);

            assign y_next[EVEN*LANE_W +: LANE_W] = res_even;
            assign y_next[ODD*LANE_W  +: LANE_W] = res_odd;
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out  <= '0;
            done_o <= 1'b0;
            busy_q <= 1'b0;
        end else begin
            done_o <= start_i;
            if (start_i) begin
                y_out  <= y_next;
                busy_q <= 1'b1;
            end else begin
                busy_q <= 1'b0;
            end
        end
    end

    assign busy_o = start_i || busy_q;

endmodule
