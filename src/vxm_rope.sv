`timescale 1ns/1ps

// RoPE row transform for VXM.
// Data lanes are FP32. Cos/sin lanes are stored as FP8 E5M2 and widened to
// FP32 before the rotate math:
//   y_even = x_even*cos - x_odd*sin
//   y_odd  = x_even*sin + x_odd*cos
module vxm_rope #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*8-1:0]      cos_fp8,
    input  logic [LANES*8-1:0]      sin_fp8,
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    output logic                    busy_o
);

    localparam int PAIRS = LANES / 2;
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_CAST_START,
        ST_CAST_WAIT,
        ST_PROD_START,
        ST_PROD_WAIT,
        ST_FMA_START,
        ST_FMA_WAIT,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [LANES*LANE_W-1:0] x_reg;
    logic [LANES*8-1:0]      cos_reg;
    logic [LANES*8-1:0]      sin_reg;

    logic [PAIRS-1:0] cast_cos_done;
    logic [PAIRS-1:0] cast_sin_done;
    logic [PAIRS-1:0] prod_cos_done;
    logic [PAIRS-1:0] prod_sin_done;
    logic [PAIRS-1:0] fma_even_done;
    logic [PAIRS-1:0] fma_odd_done;

    logic [31:0] cos_fp32 [0:PAIRS-1];
    logic [31:0] sin_fp32 [0:PAIRS-1];
    logic [31:0] x_odd_cos [0:PAIRS-1];
    logic [31:0] x_odd_sin [0:PAIRS-1];
    logic [31:0] y_even [0:PAIRS-1];
    logic [31:0] y_odd [0:PAIRS-1];

    logic cast_start;
    logic prod_start;
    logic fma_start;
    logic cast_done_all;
    logic prod_done_all;
    logic fma_done_all;

    assign cast_start = (state_q == ST_CAST_START);
    assign prod_start = (state_q == ST_PROD_START);
    assign fma_start  = (state_q == ST_FMA_START);

    assign cast_done_all = (&cast_cos_done) && (&cast_sin_done);
    assign prod_done_all = (&prod_cos_done) && (&prod_sin_done);
    assign fma_done_all  = (&fma_even_done) && (&fma_odd_done);

    generate
        for (genvar pair = 0; pair < PAIRS; pair++) begin : g_rope_pairs
            localparam int EVEN_LANE = pair * 2;
            localparam int ODD_LANE  = pair * 2 + 1;

            logic [31:0] x_even;
            logic [31:0] x_odd;
            logic [31:0] neg_x_odd_sin;

            assign x_even = x_reg[EVEN_LANE*LANE_W +: LANE_W];
            assign x_odd  = x_reg[ODD_LANE*LANE_W +: LANE_W];
            assign neg_x_odd_sin = {~x_odd_sin[pair][31], x_odd_sin[pair][30:0]};

            cvfpu_fp8_to_fp32_cast u_cos_cast (
                .clk_i      (clk),
                .rst_ni     (rst_n),
                .start_i    (cast_start),
                .fp8_bits_i (cos_reg[EVEN_LANE*8 +: 8]),
                .result_o   (cos_fp32[pair]),
                .done_o     (cast_cos_done[pair]),
                .busy_o     (/* unused */)
            );

            cvfpu_fp8_to_fp32_cast u_sin_cast (
                .clk_i      (clk),
                .rst_ni     (rst_n),
                .start_i    (cast_start),
                .fp8_bits_i (sin_reg[EVEN_LANE*8 +: 8]),
                .result_o   (sin_fp32[pair]),
                .done_o     (cast_sin_done[pair]),
                .busy_o     (/* unused */)
            );

            cvfpu_fp32_fma u_x_odd_cos (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (prod_start),
                .multiplicand_i (x_odd),
                .multiplier_i   (cos_fp32[pair]),
                .addend_i       (FP32_ZERO),
                .result_o       (x_odd_cos[pair]),
                .done_o         (prod_cos_done[pair]),
                .busy_o         (/* unused */)
            );

            cvfpu_fp32_fma u_x_odd_sin (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (prod_start),
                .multiplicand_i (x_odd),
                .multiplier_i   (sin_fp32[pair]),
                .addend_i       (FP32_ZERO),
                .result_o       (x_odd_sin[pair]),
                .done_o         (prod_sin_done[pair]),
                .busy_o         (/* unused */)
            );

            cvfpu_fp32_fma u_y_even (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (fma_start),
                .multiplicand_i (x_even),
                .multiplier_i   (cos_fp32[pair]),
                .addend_i       (neg_x_odd_sin),
                .result_o       (y_even[pair]),
                .done_o         (fma_even_done[pair]),
                .busy_o         (/* unused */)
            );

            cvfpu_fp32_fma u_y_odd (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (fma_start),
                .multiplicand_i (x_even),
                .multiplier_i   (sin_fp32[pair]),
                .addend_i       (x_odd_cos[pair]),
                .result_o       (y_odd[pair]),
                .done_o         (fma_odd_done[pair]),
                .busy_o         (/* unused */)
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            x_reg   <= '0;
            cos_reg <= '0;
            sin_reg <= '0;
            y_out   <= '0;
            done_o  <= 1'b0;
        end else begin
            done_o <= 1'b0;

            unique case (state_q)
                ST_IDLE: begin
                    if (start_i) begin
                        x_reg   <= x_in;
                        cos_reg <= cos_fp8;
                        sin_reg <= sin_fp8;
                        state_q <= ST_CAST_START;
                    end
                end

                ST_CAST_START: begin
                    state_q <= ST_CAST_WAIT;
                end

                ST_CAST_WAIT: begin
                    if (cast_done_all)
                        state_q <= ST_PROD_START;
                end

                ST_PROD_START: begin
                    state_q <= ST_PROD_WAIT;
                end

                ST_PROD_WAIT: begin
                    if (prod_done_all)
                        state_q <= ST_FMA_START;
                end

                ST_FMA_START: begin
                    state_q <= ST_FMA_WAIT;
                end

                ST_FMA_WAIT: begin
                    if (fma_done_all) begin
                        for (int pair = 0; pair < PAIRS; pair++) begin
                            y_out[(pair*2)*LANE_W +: LANE_W]     <= y_even[pair];
                            y_out[(pair*2 + 1)*LANE_W +: LANE_W] <= y_odd[pair];
                        end
                        done_o  <= 1'b1;
                        state_q <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

    assign busy_o = (state_q != ST_IDLE) || start_i;

endmodule
