`timescale 1ns/1ps

module rmsnorm #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta,
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    output logic                    busy_o
);

    localparam int ROW_W = LANES * LANE_W;
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;
    localparam logic [31:0] FP32_ONE  = 32'h3f80_0000;
    localparam logic [31:0] FP32_EPS  = 32'h3727_c5ac; // 1.0e-5

    function automatic logic [31:0] lanes_as_fp32();
        begin
            unique case (LANES)
                1:       lanes_as_fp32 = 32'h3f80_0000;
                2:       lanes_as_fp32 = 32'h4000_0000;
                4:       lanes_as_fp32 = 32'h4080_0000;
                8:       lanes_as_fp32 = 32'h4100_0000;
                16:      lanes_as_fp32 = 32'h4180_0000;
                default: lanes_as_fp32 = 32'h4100_0000;
            endcase
        end
    endfunction

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_SQUARE,
        ST_SQUARE_WAIT,
        ST_SUM1,
        ST_SUM1_WAIT,
        ST_SUM2,
        ST_SUM2_WAIT,
        ST_SUM3,
        ST_SUM3_WAIT,
        ST_DIV,
        ST_DIV_WAIT,
        ST_EPS,
        ST_EPS_WAIT,
        ST_SQRT,
        ST_SQRT_WAIT,
        ST_INV,
        ST_INV_WAIT,
        ST_MUL_INV,
        ST_MUL_INV_WAIT,
        ST_MUL_GAMMA,
        ST_MUL_GAMMA_WAIT,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [ROW_W-1:0] x_reg;
    logic [ROW_W-1:0] gamma_reg;
    logic [ROW_W-1:0] square_word;
    logic [ROW_W-1:0] norm_word;
    logic [ROW_W-1:0] scaled_word;
    logic [LANES-1:0] square_done;
    logic [LANES-1:0] mul_inv_done;
    logic [LANES-1:0] mul_gamma_done;

    logic [3:0][31:0] sum1_result;
    logic [3:0]       sum1_done;
    logic [1:0][31:0] sum2_result;
    logic [1:0]       sum2_done;
    logic [31:0]      sum3_result;
    logic             sum3_done;
    logic [31:0]      mean_square;
    logic             div_done;
    logic [31:0]      mean_square_eps;
    logic             eps_done;
    logic [31:0]      rms_value;
    logic             sqrt_done;
    logic [31:0]      inv_rms;
    logic             inv_done;

    logic launch_square;
    logic launch_sum1;
    logic launch_sum2;
    logic launch_sum3;
    logic launch_div;
    logic launch_eps;
    logic launch_sqrt;
    logic launch_inv;
    logic launch_mul_inv;
    logic launch_mul_gamma;

    assign launch_square    = (state_q == ST_SQUARE);
    assign launch_sum1      = (state_q == ST_SUM1);
    assign launch_sum2      = (state_q == ST_SUM2);
    assign launch_sum3      = (state_q == ST_SUM3);
    assign launch_div       = (state_q == ST_DIV);
    assign launch_eps       = (state_q == ST_EPS);
    assign launch_sqrt      = (state_q == ST_SQRT);
    assign launch_inv       = (state_q == ST_INV);
    assign launch_mul_inv   = (state_q == ST_MUL_INV);
    assign launch_mul_gamma = (state_q == ST_MUL_GAMMA);

    generate
        for (genvar i = 0; i < LANES; i++) begin : g_rms_lanes
            logic [31:0] x_lane;
            logic [31:0] gamma_lane;
            logic [31:0] square_lane;
            logic [31:0] norm_lane;
            logic [31:0] scaled_lane;

            assign x_lane     = x_reg[i*LANE_W +: 32];
            assign gamma_lane = gamma_reg[i*LANE_W +: 32];
            assign square_word[i*LANE_W +: 32] = square_lane;
            assign norm_word[i*LANE_W +: 32]   = norm_lane;
            assign scaled_word[i*LANE_W +: 32] = scaled_lane;

            cvfpu_fp32_fma u_square (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (launch_square),
                .multiplicand_i (x_lane),
                .multiplier_i   (x_lane),
                .addend_i       (FP32_ZERO),
                .result_o       (square_lane),
                .done_o         (square_done[i]),
                .busy_o         (/* unused */)
            );

            cvfpu_fp32_fma u_mul_inv (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (launch_mul_inv),
                .multiplicand_i (x_lane),
                .multiplier_i   (inv_rms),
                .addend_i       (FP32_ZERO),
                .result_o       (norm_lane),
                .done_o         (mul_inv_done[i]),
                .busy_o         (/* unused */)
            );

            cvfpu_fp32_fma u_mul_gamma (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (launch_mul_gamma),
                .multiplicand_i (norm_word[i*LANE_W +: 32]),
                .multiplier_i   (gamma_lane),
                .addend_i       (FP32_ZERO),
                .result_o       (scaled_lane),
                .done_o         (mul_gamma_done[i]),
                .busy_o         (/* unused */)
            );
        end

        for (genvar j = 0; j < 4; j++) begin : g_sum1
            cvfpu_fp32_addsub u_sum1 (
                .clk_i    (clk),
                .rst_ni   (rst_n),
                .start_i  (launch_sum1),
                .sub_i    (1'b0),
                .a_i      (square_word[(2*j)*LANE_W +: 32]),
                .b_i      (square_word[(2*j+1)*LANE_W +: 32]),
                .result_o (sum1_result[j]),
                .done_o   (sum1_done[j]),
                .busy_o   (/* unused */)
            );
        end

        for (genvar m = 0; m < 2; m++) begin : g_sum2
            cvfpu_fp32_addsub u_sum2 (
                .clk_i    (clk),
                .rst_ni   (rst_n),
                .start_i  (launch_sum2),
                .sub_i    (1'b0),
                .a_i      (sum1_result[2*m]),
                .b_i      (sum1_result[2*m+1]),
                .result_o (sum2_result[m]),
                .done_o   (sum2_done[m]),
                .busy_o   (/* unused */)
            );
        end
    endgenerate

    cvfpu_fp32_addsub u_sum3 (
        .clk_i    (clk),
        .rst_ni   (rst_n),
        .start_i  (launch_sum3),
        .sub_i    (1'b0),
        .a_i      (sum2_result[0]),
        .b_i      (sum2_result[1]),
        .result_o (sum3_result),
        .done_o   (sum3_done),
        .busy_o   (/* unused */)
    );

    cvfpu_fp32_div u_div_lanes (
        .clk_i      (clk),
        .rst_ni     (rst_n),
        .start_i    (launch_div),
        .dividend_i (sum3_result),
        .divisor_i  (lanes_as_fp32()),
        .result_o   (mean_square),
        .done_o     (div_done),
        .busy_o     (/* unused */)
    );

    cvfpu_fp32_addsub u_add_eps (
        .clk_i    (clk),
        .rst_ni   (rst_n),
        .start_i  (launch_eps),
        .sub_i    (1'b0),
        .a_i      (mean_square),
        .b_i      (FP32_EPS),
        .result_o (mean_square_eps),
        .done_o   (eps_done),
        .busy_o   (/* unused */)
    );

    cvfpu_fp32_sqrt u_sqrt (
        .clk_i    (clk),
        .rst_ni   (rst_n),
        .start_i  (launch_sqrt),
        .a_i      (mean_square_eps),
        .result_o (rms_value),
        .done_o   (sqrt_done),
        .busy_o   (/* unused */)
    );

    cvfpu_fp32_div u_inv (
        .clk_i      (clk),
        .rst_ni     (rst_n),
        .start_i    (launch_inv),
        .dividend_i (FP32_ONE),
        .divisor_i  (rms_value),
        .result_o   (inv_rms),
        .done_o     (inv_done),
        .busy_o     (/* unused */)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            x_reg   <= '0;
            gamma_reg <= '0;
            y_out   <= '0;
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    if (start_i) begin
                        x_reg     <= x_in;
                        gamma_reg <= gamma;
                        state_q   <= ST_SQUARE;
                    end
                end
                ST_SQUARE:        state_q <= ST_SQUARE_WAIT;
                ST_SQUARE_WAIT:   if (&square_done) state_q <= ST_SUM1;
                ST_SUM1:          state_q <= ST_SUM1_WAIT;
                ST_SUM1_WAIT:     if (&sum1_done) state_q <= ST_SUM2;
                ST_SUM2:          state_q <= ST_SUM2_WAIT;
                ST_SUM2_WAIT:     if (&sum2_done) state_q <= ST_SUM3;
                ST_SUM3:          state_q <= ST_SUM3_WAIT;
                ST_SUM3_WAIT:     if (sum3_done) state_q <= ST_DIV;
                ST_DIV:           state_q <= ST_DIV_WAIT;
                ST_DIV_WAIT:      if (div_done) state_q <= ST_EPS;
                ST_EPS:           state_q <= ST_EPS_WAIT;
                ST_EPS_WAIT:      if (eps_done) state_q <= ST_SQRT;
                ST_SQRT:          state_q <= ST_SQRT_WAIT;
                ST_SQRT_WAIT:     if (sqrt_done) state_q <= ST_INV;
                ST_INV:           state_q <= ST_INV_WAIT;
                ST_INV_WAIT:      if (inv_done) state_q <= ST_MUL_INV;
                ST_MUL_INV:       state_q <= ST_MUL_INV_WAIT;
                ST_MUL_INV_WAIT:  if (&mul_inv_done) state_q <= ST_MUL_GAMMA;
                ST_MUL_GAMMA:     state_q <= ST_MUL_GAMMA_WAIT;
                ST_MUL_GAMMA_WAIT: begin
                    if (&mul_gamma_done) begin
                        y_out   <= scaled_word;
                        state_q <= ST_DONE;
                    end
                end
                ST_DONE:          state_q <= ST_IDLE;
                default:          state_q <= ST_IDLE;
            endcase
        end
    end

    assign done_o = (state_q == ST_DONE);
    assign busy_o = (state_q != ST_IDLE);

    logic unused_beta;
    assign unused_beta = ^beta;

endmodule
