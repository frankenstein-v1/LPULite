`timescale 1ns/1ns

import lpu_pkg::*;

// Handoff note:
// - Current router behavior is preserved.
// - Transpose implementation is intentionally not finished here.
// - Start in section 2 ("TODO: transpose mode state goes here").
// - Then update section 5 to switch between normal router output and
//   transpose-emission output.

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
    // 2. TODO: transpose mode state goes here
    // -------------------------------------------------------------------------
    //
    // First real transpose state to add:
    //
    // superlane_t transpose_rows [0:3];
    // logic [1:0] transpose_load_idx;
    // logic [1:0] transpose_emit_idx;
    // logic       transpose_loading;
    // logic       transpose_emitting;
    //
    // Mental model:
    // - LOAD phase: capture 4 incoming rows into transpose_rows[0..3]
    // - EMIT phase: output 4 transposed rows, one per cycle
    //
    // Keep transpose logic separate from the current router path. Do not try to
    // force transpose into the opcode case statements directly.
    //

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
    // 5. Current router outputs
    // -------------------------------------------------------------------------
    //
    // This is the existing normal SXM behavior. When you add transpose mode,
    // this is where you will eventually choose:
    //
    // if (transpose_emitting) begin
    //     eastbound_out = transpose_emit_row;
    //     westbound_out = transpose_emit_row;
    // end else begin
    //     // current router behavior below
    // end
    //
    always_comb begin
        eastbound_out = '0;
        westbound_out = '0;

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

endmodule
