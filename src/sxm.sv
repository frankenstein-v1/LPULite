`timescale 1ns/1ns

import lpu_pkg::*;

// Switch Execution Module (SXM)

module sxm (
    input  logic clk,
    input  logic rst_n,
    input  logic [11:0] opcode_input,
    input  logic [11:0] opcode_weight,
    input  logic        load_from_west,
    input  superlane_t eastbound_in,
    output superlane_t eastbound_out,
    input  superlane_t westbound_in,
    output superlane_t westbound_out,
    output logic       emit_valid
);

    localparam logic [11:0] OP_TRANSPOSE_LOAD = 12'h5A5;
    localparam logic [11:0] OP_TRANSPOSE_EMIT = 12'hA5A;

    superlane_t transpose_rows [0:3];
    logic [1:0] transpose_load_idx;
    logic [1:0] transpose_emit_idx;
    logic       transpose_loading;
    logic       transpose_emitting;

    logic [1:0] current_emit_idx;
    logic       emit_now;
    superlane_t transpose_emit_row;
    logic       load_from_west_reg;

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
            load_from_west_reg <= 1'b0;
        end else begin
            if (opcode_input == OP_TRANSPOSE_LOAD) begin
                if (load_from_west)
                    transpose_rows[0] <= westbound_in;
                else
                    transpose_rows[0] <= eastbound_in;
                transpose_load_idx <= 2'd0;
                transpose_loading  <= 1'b1;
                load_from_west_reg <= load_from_west;
            end else if (transpose_loading) begin
                if (load_from_west_reg)
                    transpose_rows[transpose_load_idx + 2'd1] <= westbound_in;
                else
                    transpose_rows[transpose_load_idx + 2'd1] <= eastbound_in;
                if (transpose_load_idx == 2'd2) begin
                    transpose_loading <= 1'b0;
                end
                transpose_load_idx <= transpose_load_idx + 2'd1;
            end

            if (opcode_input == OP_TRANSPOSE_EMIT) begin
                transpose_emit_idx <= 2'd1;
                transpose_emitting <= 1'b1;
            end else if (transpose_emitting) begin
                if (transpose_emit_idx == 2'd3) begin
                    transpose_emitting <= 1'b0;
                end
                transpose_emit_idx <= transpose_emit_idx + 2'd1;
            end
        end
    end

    assign emit_now = transpose_emitting || (opcode_input == OP_TRANSPOSE_EMIT);
    assign current_emit_idx = (opcode_input == OP_TRANSPOSE_EMIT) ? 2'd0 : transpose_emit_idx;

    always_comb begin
        transpose_emit_row = '0;
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

    assign eastbound_out = emit_now ? transpose_emit_row : '0;
    assign westbound_out = emit_now ? transpose_emit_row : '0;
    assign emit_valid = emit_now;

    // Preserve port shape while making the reduced module intent explicit.
    logic _unused_ok;
    assign _unused_ok = ^opcode_weight;

endmodule
