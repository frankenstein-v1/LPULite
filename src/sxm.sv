`timescale 1ns/1ns

`include "lpu_pkg.sv"

// Switch Execution Module (SXM)
// Supports byte-lane routing/delay mode and 4x4/8x8 FP8 transpose mode.

module sxm #(
    parameter int LANES = 4
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

    superlane_t input_d1,  input_d2,  input_d3;
    superlane_t weight_d1, weight_d2, weight_d3;

    fp8_t transpose_rows8 [0:LANES-1][0:LANES-1];
    logic [2:0] transpose_load_idx;
    logic [2:0] transpose_emit_idx;
    logic       transpose_loading;
    logic       transpose_emitting;
    logic       load_from_west_reg;

    logic transpose_load_pulse;
    logic transpose_emit_pulse;

    assign transpose_load_pulse = (opcode_input == OP_TRANSPOSE_LOAD);
    assign transpose_emit_pulse = (opcode_input == OP_TRANSPOSE_EMIT);

    function automatic fp8_t lane_at(
        input superlane_t row,
        input logic [2:0] lane
    );
        begin
            case (lane)
                3'd0: lane_at = row[7:0];
                3'd1: lane_at = row[15:8];
                3'd2: lane_at = row[23:16];
                3'd3: lane_at = row[31:24];
                3'd4: lane_at = row[39:32];
                3'd5: lane_at = row[47:40];
                3'd6: lane_at = row[55:48];
                3'd7: lane_at = row[63:56];
                default: lane_at = 8'd0;
            endcase
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < LANES; r++) begin
                for (int c = 0; c < LANES; c++) begin
                    transpose_rows8[r][c] <= '0;
                end
            end
            transpose_load_idx <= '0;
            transpose_emit_idx <= '0;
            transpose_loading  <= 1'b0;
            transpose_emitting <= 1'b0;
            load_from_west_reg <= 1'b0;
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

            if (transpose_load_pulse) begin
                transpose_loading  <= 1'b1;
                transpose_load_idx <= '0;
                load_from_west_reg <= load_from_west;
                for (int c = 0; c < LANES; c++) begin
                    transpose_rows8[0][c] <= lane_at(load_from_west ? westbound_in : eastbound_in, c[2:0]);
                end
            end else if (transpose_loading) begin
                for (int c = 0; c < LANES; c++) begin
                    transpose_rows8[transpose_load_idx + 1'b1][c] <= lane_at(load_from_west_reg ? westbound_in : eastbound_in, c[2:0]);
                end
                transpose_load_idx <= transpose_load_idx + 1'b1;
                if (transpose_load_idx == (LANES - 2))
                    transpose_loading <= 1'b0;
            end

            if (transpose_emit_pulse) begin
                transpose_emitting <= 1'b1;
                transpose_emit_idx <= '0;
            end else if (transpose_emitting) begin
                transpose_emit_idx <= transpose_emit_idx + 1'b1;
                if (transpose_emit_idx == (LANES - 1))
                    transpose_emitting <= 1'b0;
            end
        end
    end

    logic [2:0] current_emit_idx;
    assign current_emit_idx = transpose_emit_pulse ? 3'd0 :
                              transpose_emitting   ? (transpose_emit_idx + 3'd1) :
                                                     3'd0;

    logic is_emitting;
    assign is_emitting = transpose_emitting || transpose_emit_pulse;
    assign emit_valid = is_emitting;

    superlane_t transpose_emit_row;
    always_comb begin
        transpose_emit_row = '0;
        for (int i = 0; i < LANES; i++) begin
            transpose_emit_row[i*8 +: 8] = transpose_rows8[i][current_emit_idx];
        end
    end

    function automatic logic [7:0] route_byte(
        input logic [2:0] sel,
        input superlane_t live_row,
        input superlane_t delay1_row,
        input superlane_t delay2_row,
        input superlane_t delay3_row,
        input int lane_idx
    );
        begin
            if (LANES == 8) begin
                case (sel)
                    3'b000: route_byte = (lane_idx >= 4 && sel == (lane_idx % 4)) ? live_row[lane_idx*8 +: 8] : live_row[7:0];
                    3'b001: route_byte = (lane_idx >= 4 && sel == (lane_idx % 4)) ? live_row[lane_idx*8 +: 8] : live_row[15:8];
                    3'b010: route_byte = (lane_idx >= 4 && sel == (lane_idx % 4)) ? live_row[lane_idx*8 +: 8] : live_row[23:16];
                    3'b011: route_byte = (lane_idx >= 4 && sel == (lane_idx % 4)) ? live_row[lane_idx*8 +: 8] : live_row[31:24];
                    3'b100: route_byte = lane_at(delay1_row, lane_idx[2:0]);
                    3'b101: route_byte = lane_at(delay2_row, lane_idx[2:0]);
                    3'b110: route_byte = lane_at(delay3_row, lane_idx[2:0]);
                    default: route_byte = 8'd0;
                endcase
            end else begin
                case (sel)
                    3'b000: route_byte = live_row[7:0];
                    3'b001: route_byte = live_row[15:8];
                    3'b010: route_byte = live_row[23:16];
                    3'b011: route_byte = live_row[31:24];
                    3'b100: route_byte = lane_at(delay1_row, lane_idx[1:0]);
                    3'b101: route_byte = lane_at(delay2_row, lane_idx[1:0]);
                    3'b110: route_byte = lane_at(delay3_row, lane_idx[1:0]);
                    default: route_byte = 8'd0;
                endcase
            end
        end
    endfunction

    always_comb begin
        eastbound_out = '0;
        westbound_out = '0;

        if (is_emitting) begin
            eastbound_out = transpose_emit_row;
            westbound_out = transpose_emit_row;
        end else begin
            for (int lane = 0; lane < LANES; lane++) begin
                eastbound_out[lane*8 +: 8] = route_byte(
                    opcode_input[(lane % 4)*3 +: 3],
                    eastbound_in,
                    input_d1,
                    input_d2,
                    input_d3,
                    lane
                );

                westbound_out[lane*8 +: 8] = route_byte(
                    opcode_weight[(lane % 4)*3 +: 3],
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
