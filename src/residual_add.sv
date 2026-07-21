`timescale 1ns/1ps

// Signed 32-bit residual/vector add accumulator.
module residual_add #(
    parameter int LANES  = 8,
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
    localparam logic [2:0] OP_PASS  = 3'd0;
    localparam logic [2:0] OP_CLEAR = 3'd1;
    localparam logic [2:0] OP_LOAD  = 3'd2;
    localparam logic [2:0] OP_ADD   = 3'd3;
    localparam logic [2:0] OP_EMIT  = 3'd4;

    typedef enum logic [1:0] {
        ST_IDLE
    } state_e;

    state_e state_q;

    logic [ROW_W-1:0] acc_reg;
    logic [ROW_W-1:0] add_result_word;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_residual_lanes
            logic signed [LANE_W-1:0] acc_lane;
            logic signed [LANE_W-1:0] row_lane;
            logic signed [LANE_W-1:0] add_result_lane;

            assign acc_lane = acc_reg[lane*LANE_W +: LANE_W];
            assign row_lane = row_i[lane*LANE_W +: LANE_W];
            assign add_result_lane = acc_lane + row_lane;
            assign add_result_word[lane*LANE_W +: LANE_W] = add_result_lane;
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q     <= ST_IDLE;
            acc_reg     <= '0;
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
                                acc_reg <= add_result_word;
                                done_o  <= 1'b1;
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
