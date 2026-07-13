`timescale 1ns/1ps

// FP32 residual/vector add accumulator.
//
// This block intentionally does not quantize. It keeps an FP32 row accumulator
// local to the block and only emits FP32 rows for the surrounding datapath to
// consume. VXM can feed row_o into its existing quant stage when storage is
// required.
module residual_add #(
    parameter int LANES  = 4,
    parameter int LANE_W = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    start_i,
    input  logic [2:0]              op_i,
    input  logic [LANES*LANE_W-1:0] row_i,

    output logic                    ready_o,
    output logic                    busy_o,
    output logic                    done_o,
    output logic                    row_valid_o,
    output logic [LANES*LANE_W-1:0] row_o,
    output logic [LANES*LANE_W-1:0] acc_o
);

    localparam int ROW_W = LANES * LANE_W;
    localparam logic [31:0] FP32_ONE = 32'h3f80_0000;

    localparam logic [2:0] OP_PASS  = 3'd0;
    localparam logic [2:0] OP_CLEAR = 3'd1;
    localparam logic [2:0] OP_LOAD  = 3'd2;
    localparam logic [2:0] OP_ADD   = 3'd3;
    localparam logic [2:0] OP_EMIT  = 3'd4;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_ADD_START,
        ST_ADD_WAIT
    } state_e;

    state_e state_q;

    logic [ROW_W-1:0] acc_reg;
    logic [ROW_W-1:0] add_row_reg;
    logic [ROW_W-1:0] add_result_word;
    logic [LANES-1:0] add_done_vec;
    logic             add_start;
    logic             add_done_all;

    assign add_start = (state_q == ST_ADD_START);
    assign add_done_all = &add_done_vec;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_residual_lanes
            logic [31:0] acc_lane;
            logic [31:0] add_lane;
            logic [31:0] add_result_lane;
            logic        add_done_lane;

            assign acc_lane = acc_reg[lane*LANE_W +: LANE_W];
            assign add_lane = add_row_reg[lane*LANE_W +: LANE_W];
            assign add_result_word[lane*LANE_W +: LANE_W] = add_result_lane;
            assign add_done_vec[lane] = add_done_lane;

            cvfpu_fp32_fma u_fp32_add (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (add_start),
                .multiplicand_i (acc_lane),
                .multiplier_i   (FP32_ONE),
                .addend_i       (add_lane),
                .result_o       (add_result_lane),
                .done_o         (add_done_lane),
                .busy_o         (/* unused */)
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q     <= ST_IDLE;
            acc_reg     <= '0;
            add_row_reg <= '0;
            row_o       <= '0;
            done_o      <= 1'b0;
            row_valid_o <= 1'b0;
        end else begin
            done_o      <= 1'b0;
            row_valid_o <= 1'b0;

            unique case (state_q)
                ST_IDLE: begin
                    if (start_i) begin
                        unique case (op_i)
                            OP_PASS: begin
                                row_o       <= row_i;
                                done_o      <= 1'b1;
                                row_valid_o <= 1'b1;
                            end

                            OP_CLEAR: begin
                                acc_reg <= '0;
                                done_o  <= 1'b1;
                            end

                            OP_LOAD: begin
                                acc_reg <= row_i;
                                done_o  <= 1'b1;
                            end

                            OP_ADD: begin
                                add_row_reg <= row_i;
                                state_q     <= ST_ADD_START;
                            end

                            OP_EMIT: begin
                                row_o       <= acc_reg;
                                done_o      <= 1'b1;
                                row_valid_o <= 1'b1;
                            end

                            default: begin
                                done_o <= 1'b1;
                            end
                        endcase
                    end
                end

                ST_ADD_START: begin
                    state_q <= ST_ADD_WAIT;
                end

                ST_ADD_WAIT: begin
                    if (add_done_all) begin
                        acc_reg <= add_result_word;
                        done_o  <= 1'b1;
                        state_q <= ST_IDLE;
                    end
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

    assign ready_o = (state_q == ST_IDLE);
    assign busy_o  = (state_q != ST_IDLE);
    assign acc_o   = acc_reg;

endmodule
