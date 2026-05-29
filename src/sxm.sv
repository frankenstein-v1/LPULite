`timescale 1ns/1ns

import lpu_pkg::*;

// Switch Execution Module (SXM)
// Supports two primary modes:
// 1. Normal Router Mode: Byte-lane crossbar with 3-cycle systolic delay lines.
// 2. Transpose Mode: Pipelined 4x4 matrix transposition.

module sxm (
    input  logic clk,
    input  logic rst_n,

    // Current mode:
    // - Each 3-bit field selects one output byte lane.
    // - 000..011 pick a live input byte from lane 0..3.
    // - 100..110 pick the same byte position from a 1/2/3-cycle delayed row.
    // - 111 injects a zero byte.
    input  logic [11:0] opcode_input,
    input  logic [11:0] opcode_weight,

    input  superlane_t eastbound_in,
    output superlane_t eastbound_out,

    input  superlane_t westbound_in,
    output superlane_t westbound_out
);

    // -------------------------------------------------------------------------
    // 1. Existing SXM router state
    // -------------------------------------------------------------------------
    //
    // SXM today is a byte-lane router plus three delay stages in each direction.
    // Keep this path working while you add transpose mode later.
    //
    superlane_t input_d1,  input_d2,  input_d3;
    superlane_t weight_d1, weight_d2, weight_d3;

    // -------------------------------------------------------------------------
    // 2. Transpose mode state and logic
    // -------------------------------------------------------------------------
    localparam logic [11:0] OP_TRANSPOSE_LOAD = 12'h5A5;
    localparam logic [11:0] OP_TRANSPOSE_EMIT = 12'hA5A;

    superlane_t transpose_rows [0:3];
    logic [1:0] transpose_load_idx;
    logic [1:0] transpose_emit_idx;
    logic       transpose_loading;
    logic       transpose_emitting;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            transpose_rows[0]  <= '0;
            transpose_rows[1]  <= '0;
            transpose_rows[2]  <= '0;
            transpose_rows[3]  <= '0;
            transpose_load_idx <= '0;
            transpose_emit_idx <= '0;
            transpose_loading  <= 1'b0;
            transpose_emitting <= 1'b0;
        end else begin
            // LOAD phase logic
            if (opcode_input == OP_TRANSPOSE_LOAD) begin
                transpose_loading  <= 1'b1;
                transpose_load_idx <= 2'd0;
                transpose_rows[0]  <= eastbound_in;
            end else if (transpose_loading) begin
                transpose_rows[transpose_load_idx + 1'b1] <= eastbound_in;
                if (transpose_load_idx == 2'd2) begin
                    transpose_loading <= 1'b0;
                end
                transpose_load_idx <= transpose_load_idx + 1'b1;
            end

            // EMIT phase logic
            if (opcode_input == OP_TRANSPOSE_EMIT) begin
                transpose_emitting <= 1'b1;
                transpose_emit_idx <= 2'd1;
            end else if (transpose_emitting) begin
                if (transpose_emit_idx == 2'd3) begin
                    transpose_emitting <= 1'b0;
                end
                transpose_emit_idx <= transpose_emit_idx + 1'b1;
            end
        end
    end

    logic [1:0] current_emit_idx;
    assign current_emit_idx = (opcode_input == OP_TRANSPOSE_EMIT) ? 2'd0 : transpose_emit_idx;

    logic is_emitting;
    assign is_emitting = transpose_emitting || (opcode_input == OP_TRANSPOSE_EMIT);

    superlane_t transpose_emit_row;
    always_comb begin
        case (current_emit_idx)
            2'd0: begin
                transpose_emit_row[7:0]   = transpose_rows[0][7:0];
                transpose_emit_row[15:8]  = transpose_rows[1][7:0];
                transpose_emit_row[23:16] = transpose_rows[2][7:0];
                transpose_emit_row[31:24] = transpose_rows[3][7:0];
            end
            2'd1: begin
                transpose_emit_row[7:0]   = transpose_rows[0][15:8];
                transpose_emit_row[15:8]  = transpose_rows[1][15:8];
                transpose_emit_row[23:16] = transpose_rows[2][15:8];
                transpose_emit_row[31:24] = transpose_rows[3][15:8];
            end
            2'd2: begin
                transpose_emit_row[7:0]   = transpose_rows[0][23:16];
                transpose_emit_row[15:8]  = transpose_rows[1][23:16];
                transpose_emit_row[23:16] = transpose_rows[2][23:16];
                transpose_emit_row[31:24] = transpose_rows[3][23:16];
            end
            2'd3: begin
                transpose_emit_row[7:0]   = transpose_rows[0][31:24];
                transpose_emit_row[15:8]  = transpose_rows[1][31:24];
                transpose_emit_row[23:16] = transpose_rows[2][31:24];
                transpose_emit_row[31:24] = transpose_rows[3][31:24];
            end
            default: transpose_emit_row = '0;
        endcase
    end

    // -------------------------------------------------------------------------
    // 3. Delay-line update
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_d1  <= '0;
            input_d2  <= '0;
            input_d3  <= '0;
            weight_d1 <= '0;
            weight_d2 <= '0;
            weight_d3 <= '0;
        end else begin
            input_d1  <= eastbound_in;
            input_d2  <= input_d1;
            input_d3  <= input_d2;

            weight_d1 <= westbound_in;
            weight_d2 <= weight_d1;
            weight_d3 <= weight_d2;
        end
    end

    // -------------------------------------------------------------------------
    // 4. Byte-select helper for the current router mode
    // -------------------------------------------------------------------------
    function automatic logic [7:0] route_byte(
        input logic [2:0] sel,
        input superlane_t live_row,
        input superlane_t delay1_row,
        input superlane_t delay2_row,
        input superlane_t delay3_row,
        input int lane_idx
    );
        begin
            case (sel)
                3'b000: route_byte = live_row[7:0];
                3'b001: route_byte = live_row[15:8];
                3'b010: route_byte = live_row[23:16];
                3'b011: route_byte = live_row[31:24];
                3'b100: route_byte = delay1_row[8*lane_idx +: 8];
                3'b101: route_byte = delay2_row[8*lane_idx +: 8];
                3'b110: route_byte = delay3_row[8*lane_idx +: 8];
                default: route_byte = 8'd0;
            endcase
        end
    endfunction

    // -------------------------------------------------------------------------
    // 5. Current router outputs & Transpose emission switch
    // -------------------------------------------------------------------------
    always_comb begin
        eastbound_out = '0;
        westbound_out = '0;

        if (is_emitting) begin
            eastbound_out = transpose_emit_row;
            westbound_out = transpose_emit_row;
        end else begin
            for (int lane = 0; lane < 4; lane++) begin
                eastbound_out[lane*8 +: 8] = route_byte(
                    opcode_input[lane*3 +: 3],
                    eastbound_in,
                    input_d1,
                    input_d2,
                    input_d3,
                    lane
                );

                westbound_out[lane*8 +: 8] = route_byte(
                    opcode_weight[lane*3 +: 3],
                    westbound_in,
                    weight_d1,
                    weight_d2,
                    weight_d3,
                    lane
                );
            end
        end
    end

endmodule
