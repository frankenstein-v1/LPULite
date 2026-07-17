`timescale 1ns/1ns

`include "lpu_pkg.sv"

// Switch Execution Module (SXM)
// Transpose-only block. It captures one square tile from either bus and emits
// the transposed tile to both bus directions.

module sxm #(
    parameter int LANES = $bits(superlane_t) / $bits(fp8_t)
) (
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
    localparam int IDX_W = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam logic [IDX_W-1:0] IDX_ONE = {{(IDX_W-1){1'b0}}, 1'b1};
    localparam logic [IDX_W-1:0] LAST_LOAD_IDX = LANES - 2;
    localparam logic [IDX_W-1:0] LAST_EMIT_IDX = LANES - 1;

    fp8_t transpose_rows [0:LANES-1][0:LANES-1];
    logic [IDX_W-1:0] transpose_load_idx;
    logic [IDX_W-1:0] transpose_emit_idx;
    logic             transpose_loading;
    logic             transpose_emitting;
    logic             load_from_west_reg;

    logic transpose_load_pulse;
    logic transpose_emit_pulse;

    assign transpose_load_pulse = (opcode_input == OP_TRANSPOSE_LOAD);
    assign transpose_emit_pulse = (opcode_input == OP_TRANSPOSE_EMIT);

    task automatic capture_row(
        input logic [IDX_W-1:0] row_idx,
        input superlane_t       row_data
    );
        for (int lane = 0; lane < LANES; lane++) begin
            transpose_rows[row_idx][lane] <= row_data[lane*$bits(fp8_t) +: $bits(fp8_t)];
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int row = 0; row < LANES; row++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    transpose_rows[row][lane] <= '0;
                end
            end
            transpose_load_idx <= '0;
            transpose_emit_idx <= '0;
            transpose_loading  <= 1'b0;
            transpose_emitting <= 1'b0;
            load_from_west_reg <= 1'b0;
        end else begin
            if (transpose_load_pulse) begin
                transpose_loading  <= 1'b1;
                transpose_load_idx <= '0;
                load_from_west_reg <= load_from_west;
                capture_row('0, load_from_west ? westbound_in : eastbound_in);
            end else if (transpose_loading) begin
                capture_row(
                    transpose_load_idx + IDX_ONE,
                    load_from_west_reg ? westbound_in : eastbound_in
                );
                transpose_load_idx <= transpose_load_idx + IDX_ONE;
                if (transpose_load_idx == LAST_LOAD_IDX)
                    transpose_loading <= 1'b0;
            end

            if (transpose_emit_pulse) begin
                transpose_emitting <= 1'b1;
                transpose_emit_idx <= '0;
            end else if (transpose_emitting) begin
                transpose_emit_idx <= transpose_emit_idx + IDX_ONE;
                if (transpose_emit_idx == LAST_EMIT_IDX)
                    transpose_emitting <= 1'b0;
            end
        end
    end

    logic [IDX_W-1:0] current_emit_idx;
    assign current_emit_idx = transpose_emit_pulse ? '0 :
                              transpose_emitting   ? (transpose_emit_idx + IDX_ONE) :
                                                     '0;

    logic is_emitting;
    assign is_emitting = transpose_emitting || transpose_emit_pulse;
    assign emit_valid = is_emitting;

    superlane_t transpose_emit_row;
    always_comb begin
        transpose_emit_row = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            transpose_emit_row[lane*$bits(fp8_t) +: $bits(fp8_t)] = transpose_rows[lane][current_emit_idx];
        end
    end

    always_comb begin
        eastbound_out = '0;
        westbound_out = '0;

        if (is_emitting) begin
            eastbound_out = transpose_emit_row;
            westbound_out = transpose_emit_row;
        end
    end

    // Keep the legacy port in the interface while SXM moves to transpose-only.
    logic unused_opcode_weight;
    assign unused_opcode_weight = ^opcode_weight;

endmodule
