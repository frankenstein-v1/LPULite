// ==============================================================================
// Mathematical Core & Architecture Inspiration Credit: SuryaHead
// Repository: https://github.com/SurjaHead/softmax-in-hardware
// Completely structurally rewritten to support parallel LANES and AXI4-Stream
// for the tinyLPU architecture.
// ==============================================================================
`default_nettype none
`timescale 1ns/1ns

module softmax #(
    parameter int LANES   = 4,
    parameter int LANE_W  = 32
) (
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     in_valid,
    input  logic                     input_mode_fp,
    input  logic [LANES*LANE_W-1:0]  x_in,

    output logic                     in_ready,
    output logic                     out_valid,
    output logic                     out_mode_fp,
    output logic [LANES*LANE_W-1:0]  y_out
);

    localparam int MAX_BITS = 30;
    localparam int OUT_BITS = 8;
    localparam logic [LANE_W-1:0] RECIP_DIVIDEND = 32'd1 << MAX_BITS;
    localparam int SHIFT = MAX_BITS - OUT_BITS;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_INT_DIVIDE,
        ST_INT_DONE,
        ST_FP_MAX_L1_START,
        ST_FP_MAX_L1_WAIT,
        ST_FP_MAX_L2_START,
        ST_FP_MAX_L2_WAIT,
        ST_FP_MAX_L3_START,
        ST_FP_MAX_L3_WAIT,
        ST_FP_DELTA_START,
        ST_FP_DELTA_WAIT,
        ST_FP_EXP_CAPTURE,
        ST_FP_SUM_L1_START,
        ST_FP_SUM_L1_WAIT,
        ST_FP_SUM_L2_START,
        ST_FP_SUM_L2_WAIT,
        ST_FP_SUM_L3_START,
        ST_FP_SUM_L3_WAIT,
        ST_FP_NORM_START,
        ST_FP_NORM_WAIT,
        ST_FP_DONE
    } state_t;

    function automatic logic signed [LANE_W-1:0] fp32_to_q8_8(
        input logic [31:0] fp_bits
    );
        logic        sign_bit;
        logic [7:0]  exp_bits;
        logic [22:0] frac_bits;
        logic [23:0] significand;
        integer      exp_unbiased;
        integer      shift_amount;
        longint signed scaled_value;
        begin
            sign_bit = fp_bits[31];
            exp_bits = fp_bits[30:23];
            frac_bits = fp_bits[22:0];

            if ((exp_bits == 8'h00) && (frac_bits == 23'd0)) begin
                fp32_to_q8_8 = '0;
            end else if (exp_bits == 8'hff) begin
                fp32_to_q8_8 = sign_bit ? 32'sh8000_0000 : 32'sh7fff_ffff;
            end else begin
                if (exp_bits == 8'h00) begin
                    significand = {1'b0, frac_bits};
                    exp_unbiased = -126;
                end else begin
                    significand = {1'b1, frac_bits};
                    exp_unbiased = exp_bits - 127;
                end

                shift_amount = exp_unbiased - 23 + 8;
                scaled_value = $signed({1'b0, significand});

                if (shift_amount >= 0) begin
                    if (shift_amount > 30)
                        scaled_value = 64'sh7fff_ffff;
                    else
                        scaled_value = scaled_value <<< shift_amount;
                end else if (-shift_amount > 62) begin
                    scaled_value = 64'sd0;
                end else begin
                    scaled_value = scaled_value >>> (-shift_amount);
                end

                if (sign_bit)
                    scaled_value = -scaled_value;

                if (scaled_value > 64'sh7fff_ffff)
                    fp32_to_q8_8 = 32'sh7fff_ffff;
                else if (scaled_value < -64'sh8000_0000)
                    fp32_to_q8_8 = 32'sh8000_0000;
                else
                    fp32_to_q8_8 = scaled_value[LANE_W-1:0];
            end
        end
    endfunction

    function automatic logic [31:0] uq8_8_to_fp32(
        input logic [LANE_W-1:0] fixed_value
    );
        logic [31:0] normalized;
        logic [7:0]  exponent_bits;
        integer      msb_idx;
        begin
            if (fixed_value == '0) begin
                uq8_8_to_fp32 = 32'h0000_0000;
            end else begin
                msb_idx = 0;
                for (int idx = 0; idx < LANE_W; idx++) begin
                    if (((fixed_value >> idx) & 1'b1) != 1'b0)
                        msb_idx = idx;
                end

                exponent_bits = msb_idx + 8'd119;
                if (msb_idx <= 23)
                    normalized = fixed_value << (23 - msb_idx);
                else
                    normalized = fixed_value >> (msb_idx - 23);
                uq8_8_to_fp32 = {1'b0, exponent_bits, normalized[22:0]};
            end
        end
    endfunction

    // ---------------------------------------------------------
    // Integer softmax datapath
    // ---------------------------------------------------------
    logic signed [LANE_W-1:0] int_lane_data [0:LANES-1];
    logic signed [LANE_W-1:0] int_lane_sub  [0:LANES-1];
    logic signed [LANE_W-1:0] int_lane_exp  [0:LANES-1];
    logic signed [LANE_W-1:0] int_lane_exp_reg [0:LANES-1];
    logic signed [LANE_W-1:0] int_lane_out [0:LANES-1];
    logic signed [LANE_W-1:0] int_lane_max;
    logic signed [LANE_W-1:0] sum_exp;
    logic signed [LANE_W-1:0] sum_exp_reg;
    logic [LANE_W-1:0] quotient, remainder;
    logic divider_start;
    logic divider_done;

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            int_lane_data[i] = x_in[i*LANE_W +: LANE_W];
        end
    end

    always_comb begin
        int_lane_max = int_lane_data[0];
        for (int i = 1; i < LANES; i++) begin
            if (int_lane_data[i] > int_lane_max)
                int_lane_max = int_lane_data[i];
        end
    end

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            int_lane_sub[i] = int_lane_data[i] - int_lane_max;
        end
    end

    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_int_exp
            lut_softmax_exp #(.DW(LANE_W)) exp_inst (
                .clk(clk),
                .rst(~rst_n),
                .q(int_lane_sub[i]),
                .q_out(int_lane_exp[i])
            );
        end
    endgenerate

    always_comb begin
        sum_exp = '0;
        for (int i = 0; i < LANES; i++) begin
            sum_exp = sum_exp + int_lane_exp[i];
        end
    end

    lut_softmax_div #(.DW(LANE_W)) u_lut_softmax_div (
        .clk(clk),
        .rst(~rst_n),
        .start(divider_start),
        .dividend(RECIP_DIVIDEND),
        .divisor(sum_exp_reg),
        .quotient(quotient),
        .remainder(remainder),
        .done(divider_done)
    );

    always_comb begin
        for (int k = 0; k < LANES; k++) begin
            int_lane_out[k] = (quotient * int_lane_exp_reg[k]) >> SHIFT;
        end
    end

    // ---------------------------------------------------------
    // Floating-point softmax datapath registers & signals
    // ---------------------------------------------------------
    logic [31:0] fp_input_reg [0:LANES-1];
    
    logic [31:0] fp_max_pair0_reg;
    logic [31:0] fp_max_pair1_reg;
    logic [31:0] fp_max_pair2_reg;
    logic [31:0] fp_max_pair3_reg;
    logic [31:0] fp_max_quad0_reg;
    logic [31:0] fp_max_quad1_reg;
    logic [31:0] fp_max_reg;
    
    logic [31:0] fp_delta_reg [0:LANES-1];
    logic [31:0] fp_exp_reg   [0:LANES-1];
    logic [31:0] fp_prob_reg  [0:LANES-1];
    
    logic [31:0] fp_sum_pair0_reg;
    logic [31:0] fp_sum_pair1_reg;
    logic [31:0] fp_sum_pair2_reg;
    logic [31:0] fp_sum_pair3_reg;
    logic [31:0] fp_sum_quad0_reg;
    logic [31:0] fp_sum_quad1_reg;
    logic [31:0] fp_sum_reg;

    logic signed [LANE_W-1:0] fp_delta_q8_8 [0:LANES-1];
    logic signed [LANE_W-1:0] fp_scaled_exp [0:LANES-1];
    logic [31:0]              fp_exp_wire   [0:LANES-1];

    logic fp_cmp_l1_start;
    logic fp_cmp_l2_start;
    logic fp_cmp_l3_start;
    logic fp_delta_start;
    logic fp_sum_l1_start;
    logic fp_sum_l2_start;
    logic fp_sum_l3_start;
    logic fp_norm_start;

    logic fp_cmp01_done, fp_cmp23_done, fp_cmp45_done, fp_cmp67_done;
    logic fp_cmp01_result, fp_cmp23_result, fp_cmp45_result, fp_cmp67_result;

    logic fp_cmp_q0_done, fp_cmp_q1_done;
    logic fp_cmp_q0_result, fp_cmp_q1_result;

    logic fp_cmp_oct_done;
    logic fp_cmp_oct_result;

    logic [31:0] fp_delta_result [0:LANES-1];
    logic [LANES-1:0] fp_delta_done;

    logic [31:0] fp_sum_pair0_result, fp_sum_pair1_result, fp_sum_pair2_result, fp_sum_pair3_result;
    logic        fp_sum_pair0_done, fp_sum_pair1_done, fp_sum_pair2_done, fp_sum_pair3_done;

    logic [31:0] fp_sum_quad0_result, fp_sum_quad1_result;
    logic        fp_sum_quad0_done, fp_sum_quad1_done;

    logic [31:0] fp_sum_oct_result;
    logic        fp_sum_oct_done;

    logic [31:0] fp_norm_result [0:LANES-1];
    logic [LANES-1:0] fp_norm_done;

    logic [31:0] fp_max_pair0_next;
    logic [31:0] fp_max_pair1_next;
    logic [31:0] fp_max_pair2_next;
    logic [31:0] fp_max_pair3_next;
    logic [31:0] fp_max_quad0_next;
    logic [31:0] fp_max_quad1_next;
    logic [31:0] fp_max_oct_next;

    assign fp_max_pair0_next = fp_cmp01_result ? fp_input_reg[1] : fp_input_reg[0];
    assign fp_max_pair1_next = fp_cmp23_result ? fp_input_reg[3] : fp_input_reg[2];
    assign fp_max_pair2_next = (LANES == 8) ? (fp_cmp45_result ? fp_input_reg[5] : fp_input_reg[4]) : 32'h0;
    assign fp_max_pair3_next = (LANES == 8) ? (fp_cmp67_result ? fp_input_reg[7] : fp_input_reg[6]) : 32'h0;
    
    assign fp_max_quad0_next = fp_cmp_q0_result ? fp_max_pair1_reg : fp_max_pair0_reg;
    assign fp_max_quad1_next = (LANES == 8) ? (fp_cmp_q1_result ? fp_max_pair3_reg : fp_max_pair2_reg) : 32'h0;
    
    assign fp_max_oct_next   = fp_cmp_oct_result ? fp_max_quad1_reg : fp_max_quad0_reg;

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            fp_delta_q8_8[i] = fp32_to_q8_8(fp_delta_reg[i]);
            fp_exp_wire[i]   = uq8_8_to_fp32(fp_scaled_exp[i]);
        end
    end

    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_fp_exp
            lut_softmax_exp #(.DW(LANE_W)) exp_inst (
                .clk(clk),
                .rst(~rst_n),
                .q(fp_delta_q8_8[i]),
                .q_out(fp_scaled_exp[i])
            );
        end
    endgenerate

    // ---------------------------------------------------------
    // Dual-Lane Logic Tree Allocations
    // ---------------------------------------------------------
    generate
        if (LANES == 8) begin : gen_tree_8
            // Level 1 Max Comparators
            cvfpu_fp32_cmp u_fp_cmp01(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l1_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_input_reg[0]), .b_i(fp_input_reg[1]), .result_o(fp_cmp01_result), .done_o(fp_cmp01_done), .busy_o());
            cvfpu_fp32_cmp u_fp_cmp23(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l1_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_input_reg[2]), .b_i(fp_input_reg[3]), .result_o(fp_cmp23_result), .done_o(fp_cmp23_done), .busy_o());
            cvfpu_fp32_cmp u_fp_cmp45(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l1_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_input_reg[4]), .b_i(fp_input_reg[5]), .result_o(fp_cmp45_result), .done_o(fp_cmp45_done), .busy_o());
            cvfpu_fp32_cmp u_fp_cmp67(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l1_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_input_reg[6]), .b_i(fp_input_reg[7]), .result_o(fp_cmp67_result), .done_o(fp_cmp67_done), .busy_o());

            // Level 2 Max Comparators
            cvfpu_fp32_cmp u_fp_cmp_q0(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l2_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_max_pair0_reg), .b_i(fp_max_pair1_reg), .result_o(fp_cmp_q0_result), .done_o(fp_cmp_q0_done), .busy_o());
            cvfpu_fp32_cmp u_fp_cmp_q1(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l2_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_max_pair2_reg), .b_i(fp_max_pair3_reg), .result_o(fp_cmp_q1_result), .done_o(fp_cmp_q1_done), .busy_o());

            // Level 3 Max Comparator
            cvfpu_fp32_cmp u_fp_cmp_oct(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l3_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_max_quad0_reg), .b_i(fp_max_quad1_reg), .result_o(fp_cmp_oct_result), .done_o(fp_cmp_oct_done), .busy_o());

            // Level 1 Exponent Sums
            cvfpu_fp32_addsub u_fp_sum_pair0(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l1_start), .sub_i(1'b0), .a_i(fp_exp_reg[0]), .b_i(fp_exp_reg[1]), .result_o(fp_sum_pair0_result), .done_o(fp_sum_pair0_done), .busy_o());
            cvfpu_fp32_addsub u_fp_sum_pair1(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l1_start), .sub_i(1'b0), .a_i(fp_exp_reg[2]), .b_i(fp_exp_reg[3]), .result_o(fp_sum_pair1_result), .done_o(fp_sum_pair1_done), .busy_o());
            cvfpu_fp32_addsub u_fp_sum_pair2(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l1_start), .sub_i(1'b0), .a_i(fp_exp_reg[4]), .b_i(fp_exp_reg[5]), .result_o(fp_sum_pair2_result), .done_o(fp_sum_pair2_done), .busy_o());
            cvfpu_fp32_addsub u_fp_sum_pair3(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l1_start), .sub_i(1'b0), .a_i(fp_exp_reg[6]), .b_i(fp_exp_reg[7]), .result_o(fp_sum_pair3_result), .done_o(fp_sum_pair3_done), .busy_o());

            // Level 2 Exponent Sums
            cvfpu_fp32_addsub u_fp_sum_quad0(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l2_start), .sub_i(1'b0), .a_i(fp_sum_pair0_reg), .b_i(fp_sum_pair1_reg), .result_o(fp_sum_quad0_result), .done_o(fp_sum_quad0_done), .busy_o());
            cvfpu_fp32_addsub u_fp_sum_quad1(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l2_start), .sub_i(1'b0), .a_i(fp_sum_pair2_reg), .b_i(fp_sum_pair3_reg), .result_o(fp_sum_quad1_result), .done_o(fp_sum_quad1_done), .busy_o());

            // Level 3 Exponent Sum
            cvfpu_fp32_addsub u_fp_sum_oct(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l3_start), .sub_i(1'b0), .a_i(fp_sum_quad0_reg), .b_i(fp_sum_quad1_reg), .result_o(fp_sum_oct_result), .done_o(fp_sum_oct_done), .busy_o());
        end else begin : gen_tree_4
            // Level 1 Max Comparators
            cvfpu_fp32_cmp u_fp_cmp01(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l1_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_input_reg[0]), .b_i(fp_input_reg[1]), .result_o(fp_cmp01_result), .done_o(fp_cmp01_done), .busy_o());
            cvfpu_fp32_cmp u_fp_cmp23(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l1_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_input_reg[2]), .b_i(fp_input_reg[3]), .result_o(fp_cmp23_result), .done_o(fp_cmp23_done), .busy_o());

            // Level 2 Max Comparator
            cvfpu_fp32_cmp u_fp_cmp_q0(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_cmp_l2_start), .cmp_mode_i(2'b0), .invert_i(1'b0), .a_i(fp_max_pair0_reg), .b_i(fp_max_pair1_reg), .result_o(fp_cmp_q0_result), .done_o(fp_cmp_q0_done), .busy_o());

            // Level 1 Exponent Sums
            cvfpu_fp32_addsub u_fp_sum_pair0(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l1_start), .sub_i(1'b0), .a_i(fp_exp_reg[0]), .b_i(fp_exp_reg[1]), .result_o(fp_sum_pair0_result), .done_o(fp_sum_pair0_done), .busy_o());
            cvfpu_fp32_addsub u_fp_sum_pair1(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l1_start), .sub_i(1'b0), .a_i(fp_exp_reg[2]), .b_i(fp_exp_reg[3]), .result_o(fp_sum_pair1_result), .done_o(fp_sum_pair1_done), .busy_o());

            // Level 2 Exponent Sum
            cvfpu_fp32_addsub u_fp_sum_quad0(.clk_i(clk), .rst_ni(rst_n), .start_i(fp_sum_l2_start), .sub_i(1'b0), .a_i(fp_sum_pair0_reg), .b_i(fp_sum_pair1_reg), .result_o(fp_sum_quad0_result), .done_o(fp_sum_quad0_done), .busy_o());

            // Dummies for unused 8-lane signals/ports
            assign fp_cmp45_done = 1'b1;
            assign fp_cmp67_done = 1'b1;
            assign fp_cmp45_result = 1'b0;
            assign fp_cmp67_result = 1'b0;
            assign fp_cmp_q1_done = 1'b1;
            assign fp_cmp_q1_result = 1'b0;
            assign fp_cmp_oct_done = 1'b1;
            assign fp_cmp_oct_result = 1'b0;

            assign fp_sum_pair2_result = 32'd0;
            assign fp_sum_pair3_result = 32'd0;
            assign fp_sum_pair2_done = 1'b1;
            assign fp_sum_pair3_done = 1'b1;
            assign fp_sum_quad1_result = 32'd0;
            assign fp_sum_quad1_done = 1'b1;
            assign fp_sum_oct_result = 32'd0;
            assign fp_sum_oct_done = 1'b1;
        end
    endgenerate

    // Unified FP Delat & Normalizer Generators
    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_fp_delta_inst
            cvfpu_fp32_addsub u_fp_delta (
                .clk_i(clk),
                .rst_ni(rst_n),
                .start_i(fp_delta_start),
                .sub_i(1'b1),
                .a_i(fp_input_reg[i]),
                .b_i(fp_max_reg),
                .result_o(fp_delta_result[i]),
                .done_o(fp_delta_done[i]),
                .busy_o()
            );
        end

        for (genvar i = 0; i < LANES; i++) begin : gen_fp_norm_inst
            cvfpu_fp32_div u_fp_norm (
                .clk_i(clk),
                .rst_ni(rst_n),
                .start_i(fp_norm_start),
                .dividend_i(fp_exp_reg[i]),
                .divisor_i(fp_sum_reg),
                .result_o(fp_norm_result[i]),
                .done_o(fp_norm_done[i]),
                .busy_o()
            );
        end
    endgenerate


    // ---------------------------------------------------------
    // Control
    // ---------------------------------------------------------
    state_t state, next_state;
    logic input_mode_fp_reg;

    assign in_ready = (state == ST_IDLE);
    assign out_mode_fp = input_mode_fp_reg;

    always_comb begin
        next_state      = state;
        out_valid       = 1'b0;
        divider_start   = 1'b0;
        fp_cmp_l1_start = 1'b0;
        fp_cmp_l2_start = 1'b0;
        fp_cmp_l3_start = 1'b0;
        fp_delta_start  = 1'b0;
        fp_sum_l1_start = 1'b0;
        fp_sum_l2_start = 1'b0;
        fp_sum_l3_start = 1'b0;
        fp_norm_start   = 1'b0;

        unique case (state)
            ST_IDLE: begin
                if (in_valid) begin
                    if (input_mode_fp)
                        next_state = ST_FP_MAX_L1_START;
                    else begin
                        divider_start = 1'b1;
                        next_state = ST_INT_DIVIDE;
                    end
                end
            end

            ST_INT_DIVIDE: begin
                if (divider_done)
                    next_state = ST_INT_DONE;
            end

            ST_INT_DONE: begin
                out_valid = 1'b1;
                next_state = ST_IDLE;
            end

            ST_FP_MAX_L1_START: begin
                fp_cmp_l1_start = 1'b1;
                next_state = ST_FP_MAX_L1_WAIT;
            end

            ST_FP_MAX_L1_WAIT: begin
                if (LANES == 8) begin
                    if (fp_cmp01_done && fp_cmp23_done && fp_cmp45_done && fp_cmp67_done)
                        next_state = ST_FP_MAX_L2_START;
                end else begin
                    if (fp_cmp01_done && fp_cmp23_done)
                        next_state = ST_FP_MAX_L2_START;
                end
            end

            ST_FP_MAX_L2_START: begin
                fp_cmp_l2_start = 1'b1;
                next_state = ST_FP_MAX_L2_WAIT;
            end

            ST_FP_MAX_L2_WAIT: begin
                if (LANES == 8) begin
                    if (fp_cmp_q0_done && fp_cmp_q1_done)
                        next_state = ST_FP_MAX_L3_START;
                end else begin
                    if (fp_cmp_q0_done)
                        next_state = ST_FP_DELTA_START;
                end
            end

            ST_FP_MAX_L3_START: begin
                fp_cmp_l3_start = 1'b1;
                next_state = ST_FP_MAX_L3_WAIT;
            end

            ST_FP_MAX_L3_WAIT: begin
                if (fp_cmp_oct_done)
                    next_state = ST_FP_DELTA_START;
            end

            ST_FP_DELTA_START: begin
                fp_delta_start = 1'b1;
                next_state = ST_FP_DELTA_WAIT;
            end

            ST_FP_DELTA_WAIT: begin
                if (&fp_delta_done)
                    next_state = ST_FP_EXP_CAPTURE;
            end

            ST_FP_EXP_CAPTURE: begin
                next_state = ST_FP_SUM_L1_START;
            end

            ST_FP_SUM_L1_START: begin
                fp_sum_l1_start = 1'b1;
                next_state = ST_FP_SUM_L1_WAIT;
            end

            ST_FP_SUM_L1_WAIT: begin
                if (LANES == 8) begin
                    if (fp_sum_pair0_done && fp_sum_pair1_done && fp_sum_pair2_done && fp_sum_pair3_done)
                        next_state = ST_FP_SUM_L2_START;
                end else begin
                    if (fp_sum_pair0_done && fp_sum_pair1_done)
                        next_state = ST_FP_SUM_L2_START;
                end
            end

            ST_FP_SUM_L2_START: begin
                fp_sum_l2_start = 1'b1;
                next_state = ST_FP_SUM_L2_WAIT;
            end

            ST_FP_SUM_L2_WAIT: begin
                if (LANES == 8) begin
                    if (fp_sum_quad0_done && fp_sum_quad1_done)
                        next_state = ST_FP_SUM_L3_START;
                end else begin
                    if (fp_sum_quad0_done)
                        next_state = ST_FP_NORM_START;
                end
            end

            ST_FP_SUM_L3_START: begin
                fp_sum_l3_start = 1'b1;
                next_state = ST_FP_SUM_L3_WAIT;
            end

            ST_FP_SUM_L3_WAIT: begin
                if (fp_sum_oct_done)
                    next_state = ST_FP_NORM_START;
            end

            ST_FP_NORM_START: begin
                fp_norm_start = 1'b1;
                next_state = ST_FP_NORM_WAIT;
            end

            ST_FP_NORM_WAIT: begin
                if (&fp_norm_done)
                    next_state = ST_FP_DONE;
            end

            ST_FP_DONE: begin
                out_valid = 1'b1;
                next_state = ST_IDLE;
            end

            default: next_state = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            input_mode_fp_reg <= 1'b0;
            sum_exp_reg <= '0;
            fp_max_pair0_reg <= 32'h0000_0000;
            fp_max_pair1_reg <= 32'h0000_0000;
            fp_max_pair2_reg <= 32'h0000_0000;
            fp_max_pair3_reg <= 32'h0000_0000;
            fp_max_quad0_reg <= 32'h0000_0000;
            fp_max_quad1_reg <= 32'h0000_0000;
            fp_max_reg <= 32'h0000_0000;
            fp_sum_pair0_reg <= 32'h0000_0000;
            fp_sum_pair1_reg <= 32'h0000_0000;
            fp_sum_pair2_reg <= 32'h0000_0000;
            fp_sum_pair3_reg <= 32'h0000_0000;
            fp_sum_quad0_reg <= 32'h0000_0000;
            fp_sum_quad1_reg <= 32'h0000_0000;
            fp_sum_reg <= 32'h0000_0000;
            for (int i = 0; i < LANES; i++) begin
                int_lane_exp_reg[i] <= '0;
                fp_input_reg[i] <= 32'h0000_0000;
                fp_delta_reg[i] <= 32'h0000_0000;
                fp_exp_reg[i] <= 32'h0000_0000;
                fp_prob_reg[i] <= 32'h0000_0000;
            end
        end else begin
            state <= next_state;

            if (state == ST_IDLE && in_valid) begin
                input_mode_fp_reg <= input_mode_fp;
                if (input_mode_fp) begin
                    for (int i = 0; i < LANES; i++) begin
                        fp_input_reg[i] <= x_in[i*LANE_W +: LANE_W];
                    end
                end else begin
                    sum_exp_reg <= sum_exp;
                    for (int i = 0; i < LANES; i++) begin
                        int_lane_exp_reg[i] <= int_lane_exp[i];
                    end
                end
            end

            // Max reduction L1 capture
            if (state == ST_FP_MAX_L1_WAIT) begin
                if (LANES == 8) begin
                    if (fp_cmp01_done && fp_cmp23_done && fp_cmp45_done && fp_cmp67_done) begin
                        fp_max_pair0_reg <= fp_max_pair0_next;
                        fp_max_pair1_reg <= fp_max_pair1_next;
                        fp_max_pair2_reg <= fp_max_pair2_next;
                        fp_max_pair3_reg <= fp_max_pair3_next;
                    end
                end else begin
                    if (fp_cmp01_done && fp_cmp23_done) begin
                        fp_max_pair0_reg <= fp_max_pair0_next;
                        fp_max_pair1_reg <= fp_max_pair1_next;
                    end
                end
            end

            // Max reduction L2 capture
            if (state == ST_FP_MAX_L2_WAIT) begin
                if (LANES == 8) begin
                    if (fp_cmp_q0_done && fp_cmp_q1_done) begin
                        fp_max_quad0_reg <= fp_max_quad0_next;
                        fp_max_quad1_reg <= fp_max_quad1_next;
                    end
                end else begin
                    if (fp_cmp_q0_done) begin
                        fp_max_reg <= fp_max_quad0_next;
                    end
                end
            end

            // Max reduction L3 capture
            if (state == ST_FP_MAX_L3_WAIT && fp_cmp_oct_done) begin
                fp_max_reg <= fp_max_oct_next;
            end

            if (state == ST_FP_DELTA_WAIT && (&fp_delta_done)) begin
                for (int i = 0; i < LANES; i++) begin
                    fp_delta_reg[i] <= fp_delta_result[i];
                end
            end

            if (state == ST_FP_EXP_CAPTURE) begin
                for (int i = 0; i < LANES; i++) begin
                    fp_exp_reg[i] <= fp_exp_wire[i];
                end
            end

            // Exponent sum L1 capture
            if (state == ST_FP_SUM_L1_WAIT) begin
                if (LANES == 8) begin
                    if (fp_sum_pair0_done && fp_sum_pair1_done && fp_sum_pair2_done && fp_sum_pair3_done) begin
                        fp_sum_pair0_reg <= fp_sum_pair0_result;
                        fp_sum_pair1_reg <= fp_sum_pair1_result;
                        fp_sum_pair2_reg <= fp_sum_pair2_result;
                        fp_sum_pair3_reg <= fp_sum_pair3_result;
                    end
                end else begin
                    if (fp_sum_pair0_done && fp_sum_pair1_done) begin
                        fp_sum_pair0_reg <= fp_sum_pair0_result;
                        fp_sum_pair1_reg <= fp_sum_pair1_result;
                    end
                end
            end

            // Exponent sum L2 capture
            if (state == ST_FP_SUM_L2_WAIT) begin
                if (LANES == 8) begin
                    if (fp_sum_quad0_done && fp_sum_quad1_done) begin
                        fp_sum_quad0_reg <= fp_sum_quad0_result;
                        fp_sum_quad1_reg <= fp_sum_quad1_result;
                    end
                end else begin
                    if (fp_sum_quad0_done) begin
                        fp_sum_reg <= fp_sum_quad0_result;
                    end
                end
            end

            // Exponent sum L3 capture
            if (state == ST_FP_SUM_L3_WAIT && fp_sum_oct_done) begin
                fp_sum_reg <= fp_sum_oct_result;
            end

            if (state == ST_FP_NORM_WAIT && (&fp_norm_done)) begin
                for (int i = 0; i < LANES; i++) begin
                    fp_prob_reg[i] <= fp_norm_result[i];
                end
            end
        end
    end

    always_comb begin
        y_out = '0;
        if (state == ST_INT_DONE) begin
            for (int i = 0; i < LANES; i++) begin
                y_out[i*LANE_W +: LANE_W] = int_lane_out[i];
            end
        end else if (state == ST_FP_DONE) begin
            for (int i = 0; i < LANES; i++) begin
                y_out[i*LANE_W +: LANE_W] = fp_prob_reg[i];
            end
        end
    end

endmodule
