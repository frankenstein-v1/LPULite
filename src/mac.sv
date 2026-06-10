`timescale 1ns/1ps

// Floating-point MAC skeleton for the future FP MXM path.
//
// Intended algorithm:
// 1. Latch one FP8 weight when weight_load is asserted.
// 2. On en, treat input_in and the stored weight as raw FP8 bit patterns.
// 3. Cast both FP8 operands to FP32.
// 4. Perform one FP32 fused multiply-add:
//      acc_reg = acc_reg + (input_fp32 * weight_fp32)
// 5. Hold the running FP32 accumulator in acc_reg.
//
// This module is not wired into mxm.sv yet. The current integer datapath still
// uses int_mac + acc. Once CVFPU wrappers are ready, mxm can swap to this cell
// and drop the separate integer accumulator stage.
module mac #(
    parameter int FP8_W  = 8,
    parameter int FP32_W = 32
)(
    input  logic               clk,
    input  logic               rst,
    input  logic               en,
    input  logic               clear,

    // Raw FP8 operand bit patterns.
    input  logic [FP8_W-1:0]   input_in,
    input  logic               weight_load,
    input  logic [FP8_W-1:0]   weight_value,

    // Kept for interface convenience if mxm later swaps directly to this
    // module. FP formats already carry sign, so these are ignored here.
    input  logic               input_is_signed,
    input  logic               weight_is_signed,

    // FP32 accumulator bits.
    output logic [FP32_W-1:0]  accum_out,
    output logic               result_valid,
    output logic               busy
);

    typedef enum logic [1:0] {
        MAC_IDLE,
        MAC_CAST,
        MAC_FMA,
        MAC_WRITEBACK
    } mac_state_t;

    mac_state_t state_q, state_d;

    // Stored FP8 weight and FP32 running sum.
    logic [FP8_W-1:0]  weight_reg_q, weight_reg_d;
    logic [FP32_W-1:0] acc_reg_q, acc_reg_d;

    // Snapshot the input that launched the current MAC step so the operand is
    // stable while the future CVFPU pipeline runs.
    logic [FP8_W-1:0]  input_latched_q, input_latched_d;

    logic [FP32_W-1:0] input_fp32_bits;
    logic [FP32_W-1:0] weight_fp32_bits;
    logic [FP32_W-1:0] fma_result_bits;

    logic cast_launch;
    logic input_cast_done;
    logic weight_cast_done;
    logic cast_done;
    logic fma_launch;
    logic fma_done;

    // Sign is encoded in the FP8 format itself. Keep the ports referenced so
    // lint/sim do not complain once this file is used on its own.
    logic _unused_signed_ctrl;
    assign _unused_signed_ctrl = input_is_signed ^ weight_is_signed;

    assign cast_launch = (state_q == MAC_IDLE) && en;
    assign cast_done   = input_cast_done && weight_cast_done;
    assign fma_launch  = (state_q == MAC_CAST) && cast_done;

    cvfpu_fp8_to_fp32_cast u_input_cast (
        .clk_i      (clk),
        .rst_ni     (~rst),
        .start_i    (cast_launch),
        .fp8_bits_i (input_latched_d),
        .result_o   (input_fp32_bits),
        .done_o     (input_cast_done),
        .busy_o     (/* unused */)
    );

    cvfpu_fp8_to_fp32_cast u_weight_cast (
        .clk_i      (clk),
        .rst_ni     (~rst),
        .start_i    (cast_launch),
        .fp8_bits_i (weight_reg_q),
        .result_o   (weight_fp32_bits),
        .done_o     (weight_cast_done),
        .busy_o     (/* unused */)
    );

    cvfpu_fp32_fma u_accum_fma (
        .clk_i        (clk),
        .rst_ni       (~rst),
        .start_i      (fma_launch),
        .multiplicand_i(input_fp32_bits),
        .multiplier_i (weight_fp32_bits),
        .addend_i     (acc_reg_q),
        .result_o     (fma_result_bits),
        .done_o       (fma_done),
        .busy_o       (/* unused */)
    );

    always_comb begin
        state_d = state_q;
        weight_reg_d = weight_reg_q;
        acc_reg_d = acc_reg_q;
        input_latched_d = input_latched_q;

        if (weight_load) begin
            weight_reg_d = weight_value;
        end

        if (clear) begin
            // IEEE-754 +0.0 in FP32.
            acc_reg_d = 32'h0000_0000;
        end

        unique case (state_q)
            MAC_IDLE: begin
                if (en) begin
                    input_latched_d = input_in;
                    state_d = MAC_CAST;
                end
            end

            MAC_CAST: begin
                // Step 1: convert input_latched_q and weight_reg_q from FP8 to
                // FP32 with CVFPU cast helpers.
                if (cast_done) begin
                    state_d = MAC_FMA;
                end
            end

            MAC_FMA: begin
                // Step 2: run one FP32 FMA:
                //   acc_reg_q + input_fp32_bits * weight_fp32_bits
                if (fma_done) begin
                    acc_reg_d = fma_result_bits;
                    state_d = MAC_WRITEBACK;
                end
            end

            MAC_WRITEBACK: begin
                // Step 3: accumulator has been updated. Pulse result_valid for
                // one cycle, then return to idle.
                state_d = MAC_IDLE;
            end

            default: begin
                state_d = MAC_IDLE;
            end
        endcase
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q <= MAC_IDLE;
            weight_reg_q <= '0;
            acc_reg_q <= 32'h0000_0000;
            input_latched_q <= '0;
        end else begin
            state_q <= state_d;
            weight_reg_q <= weight_reg_d;
            acc_reg_q <= acc_reg_d;
            input_latched_q <= input_latched_d;
        end
    end

    assign accum_out = acc_reg_q;
    assign busy = (state_q != MAC_IDLE);
    assign result_valid = (state_q == MAC_WRITEBACK);

endmodule
