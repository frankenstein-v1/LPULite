`timescale 1ns/1ns
`include "lpu_pkg.sv"

module vxm #(
    parameter int LANES   = 8,
    parameter int LANE_W  = 32, // Upgraded to 32 bits
    parameter int ALU_W   = 32, // Upgraded ALU width to 32 bits per user request
    parameter int RMSNORM_CHUNKS = 8,
    parameter int SOFTMAX_CHUNKS = 64
) (
    input  logic clk,
    input  logic rst_n,

    // Input vectors
    input  logic [LANES*LANE_W-1:0] stream_in_data,
    input  logic [LANES*LANE_W-1:0] stream_in_bias,
    
    // Valid/ready signaling
    input  logic                    in_valid,
    output logic                    in_ready,

    // Control bits for the Muxes (3 cascaded 2-to-1 muxes)
    // [0]: Mux1 Sel
    // [1]: Mux2 Sel
    // [2]: Mux3 Sel
    // [3]: Softmax Bypass Sel (0: Quantize, 1: Softmax)
    input  logic [3:0]              vxm_ctrl,
    input  logic                    fp_quant_mode,

    // RoPE control and FP8 cos/sin operands
    input  logic                    rope_en,
    input  logic [LANES*8-1:0]      rope_cos_fp8,
    input  logic [LANES*8-1:0]      rope_sin_fp8,

    // Residual accumulator control
    input  logic [2:0]              residual_op,
    input  logic [31:0]             scale_factor,

    // RMSNorm control and parameters
    input  logic                    rmsnorm_bypass,
    input  logic [LANES*LANE_W-1:0] rmsnorm_gamma,
    input  logic [LANES*LANE_W-1:0] rmsnorm_beta,

    // Outputs
    output logic [LANES*8-1:0] stream_out,
    output logic [31:0] stream_out_scale,
    output logic                    out_valid,
    input  logic                    out_ready
);

    localparam int ROW_W = LANES * LANE_W;
    localparam logic [2:0] RES_OP_PASS = 3'd0;
    localparam logic [2:0] RES_OP_EMIT = 3'd4;

    //stage 0 -> input registers for each lane 
    logic [ROW_W-1:0] s0_data_reg; 
    logic [ROW_W-1:0] s0_bias_reg;
    logic [3:0]       s0_ctrl_reg;
    logic             s0_valid;
    
    //stage 1-> bias add registers for each lane
    logic [ROW_W-1:0] s1_bias_reg;
    logic [2:0]       s1_ctrl_reg;
    logic             s1_valid;

    //stage 2 -> ReLU register for each lane
    logic [ROW_W-1:0] s2_relu_reg;
    logic [1:0]       s2_ctrl_reg;
    logic             s2_valid;

    //stage 3-> scale register for each lane
    logic [ROW_W-1:0] s3_scale_reg;
    logic             s3_bypass_sel_reg;
    logic             s3_valid;

    //stage 4-> softmax/quantization register
    logic [ROW_W-1:0] s4_handoff_reg;
    logic             s4_bypass_sel_reg;
    logic             s4_valid;

    logic [ROW_W-1:0] s1_bias_next; // integer bias/bypass result
    logic [ROW_W-1:0] s2_relu_next; // ReLU/bypass result
    logic [ROW_W-1:0] s3_scale_next; // integer scale/bypass result

    //outputs for quantize & softmax & muxes
    logic [LANES*8-1:0]       quantize_out;
    logic [31:0]              quantize_scale_out;
    logic                     quantize_valid;
    logic                     chunked_softmax_in_valid;
    logic                     chunked_softmax_in_ready;
    logic                     chunked_softmax_out_valid;
    logic                     chunked_softmax_out_mode_fp;
    logic [ROW_W-1:0]         chunked_softmax_out;
    logic signed [7:0]        chunked_softmax_out_scale;
    logic                     chunked_softmax_out_ready;
    logic                     chunked_softmax_busy;
    logic                     softmax_stall;
    logic                     stall_pipeline;

    logic                     softmax_active_valid;
    logic [ROW_W-1:0]         softmax_active_data;
    logic                     softmax_active_is_fp;
    logic                     softmax_result_valid;
    logic [ROW_W-1:0]         softmax_result_reg;
    logic                     softmax_result_mode_fp_reg;
    logic                     softmax_result_accept;
    logic                     softmax_result_can_take;
    logic [LANES*LANE_W-1:0]  mux_out;
    logic                     mux_valid;
    quant_mode_e              quant_mode;
    logic                     quant_issue;
    logic                     quant_inflight;
    logic                     quant_slot_available;
    logic                     rope_start;
    logic                     rope_done;
    logic                     rope_busy;
    logic                     rope_inflight;
    logic                     rope_result_valid;
    logic                     rope_stall;
    logic [ROW_W-1:0]         rope_out;
    logic [ROW_W-1:0]         rope_result_reg;
    logic [ROW_W-1:0]         pre_layernorm_in;
    logic [ROW_W-1:0]         rmsnorm_out;
    logic                     rmsnorm_start;
    logic                     rmsnorm_in_ready;
    logic                     rmsnorm_done;
    logic                     rmsnorm_out_ready;
    logic                     rmsnorm_busy;
    logic                     rmsnorm_input_valid;
    logic                     rmsnorm_output_valid;
    logic                     rmsnorm_stall;
    logic [ROW_W-1:0]         residual_row_in;
    logic [ROW_W-1:0]         residual_row_out;
    logic [ROW_W-1:0]         residual_acc_out;
    logic [ROW_W-1:0]         residual_result_reg;
    logic                     residual_start;
    logic                     residual_emit_cmd;
    logic                     residual_input_valid;
    logic                     residual_ready;
    logic                     residual_busy;
    logic                     residual_done;
    logic                     residual_row_valid;
    logic                     residual_result_valid;
    logic                     residual_stall;
    logic                     residual_active_mode_softmax;
    logic                     residual_active_fp_mode;
    logic                     residual_result_mode_softmax;
    logic                     residual_result_fp_mode;
    logic [LANES*8-1:0]       stream_out_reg;
    logic [31:0]              stream_out_scale_reg;
    logic                     stream_out_valid_reg;
    logic                     unused_scale_factor;

    assign unused_scale_factor = ^scale_factor;

    //extend from 1 lane to 4 lanes

    function automatic logic signed [ALU_W-1:0] widen_lane(
        input logic signed [LANE_W-1:0] lane_value
    );
        widen_lane = {{(ALU_W-LANE_W){lane_value[LANE_W-1]}}, lane_value};
    endfunction

    // stage buffering logic using generate loop for all lanes
    // (Using generate loops and continuous assigns to avoid iverilog compilation issues)
    generate
        genvar i;
        for (i = 0; i < LANES; i++) begin : g_vxm_lanes
            // Stage 1: bias add logic
            logic signed [LANE_W-1:0] data_lane;
            logic signed [LANE_W-1:0] bias_lane_s0;
            logic signed [ALU_W-1:0]  bias_add_out;
            logic signed [ALU_W-1:0]  mux1_out;

            assign data_lane    = s0_data_reg[i*LANE_W +: LANE_W];
            assign bias_lane_s0 = s0_bias_reg[i*LANE_W +: LANE_W];
            assign bias_add_out = widen_lane(data_lane) + widen_lane(bias_lane_s0);
            assign mux1_out     = s0_ctrl_reg[0] ? bias_add_out : widen_lane(data_lane);
            assign s1_bias_next[i*LANE_W +: LANE_W] = mux1_out[LANE_W-1:0];

            // Stage 2: ReLU logic (32-bit signed fixed-point)
            logic signed [LANE_W-1:0] bias_lane_s1;
            logic signed [ALU_W-1:0]  relu_out_int;

            assign bias_lane_s1 = s1_bias_reg[i*LANE_W +: LANE_W];
            assign relu_out_int = (widen_lane(bias_lane_s1) < 0) ? '0 : widen_lane(bias_lane_s1);
            assign s2_relu_next[i*LANE_W +: LANE_W] = s1_ctrl_reg[0] ? relu_out_int[LANE_W-1:0] : bias_lane_s1;

            // Stage 3: scale logic
            logic signed [LANE_W-1:0] relu_lane;
            logic signed [ALU_W-1:0]  scale_out_int;

            assign relu_lane    = s2_relu_reg[i*LANE_W +: LANE_W];
            assign scale_out_int = widen_lane(relu_lane) >>> 1;
            assign s3_scale_next[i*LANE_W +: LANE_W] = s2_ctrl_reg[0] ? scale_out_int[LANE_W-1:0] : relu_lane;
        end
    endgenerate
    
    assign chunked_softmax_in_valid = s4_valid && s4_bypass_sel_reg && !stall_pipeline;

    assign softmax_active_valid = softmax_result_valid;
    assign softmax_active_data  = softmax_result_reg;
    assign softmax_active_is_fp = softmax_result_mode_fp_reg;

    assign mux_out   = softmax_active_valid ? softmax_active_data : s4_handoff_reg;
    assign mux_valid = softmax_active_valid || (s4_valid && !s4_bypass_sel_reg);
    always_comb begin
        if (residual_result_mode_softmax)
            quant_mode = QUANT_SOFTMAX_U8;
        else if (residual_result_fp_mode)
            quant_mode = QUANT_FP8_E5M2;
        else
            quant_mode = QUANT_SIGNED_INT8;
    end
    assign quant_slot_available = !quant_inflight && (!stream_out_valid_reg || out_ready);
    assign quant_issue = residual_result_valid && quant_slot_available;
    assign rope_start = rope_en && mux_valid && !rope_inflight && !rope_result_valid;
    assign residual_emit_cmd = in_valid && (residual_op == RES_OP_EMIT);
    assign rmsnorm_input_valid = rope_en ? rope_result_valid : mux_valid;
    assign rmsnorm_start = !rmsnorm_bypass &&
                           rmsnorm_input_valid &&
                           rmsnorm_in_ready;
    assign rmsnorm_output_valid = rmsnorm_bypass ? rmsnorm_input_valid : rmsnorm_done;
    assign residual_input_valid = residual_emit_cmd ||
                                  rmsnorm_output_valid;
    assign residual_start = residual_input_valid &&
                            residual_ready &&
                            !residual_result_valid;
    assign softmax_result_accept = residual_start && softmax_result_valid;
    assign softmax_result_can_take = !softmax_result_valid || softmax_result_accept;
    assign chunked_softmax_out_ready = softmax_result_can_take;
    assign softmax_stall = s4_valid && s4_bypass_sel_reg && !chunked_softmax_in_ready;
    assign rope_stall = rope_en && (rope_inflight || (mux_valid && !rope_result_valid));
    assign rmsnorm_stall = !rmsnorm_bypass &&
                            rmsnorm_input_valid &&
                            !rmsnorm_in_ready;
    assign rmsnorm_out_ready = !rmsnorm_bypass && rmsnorm_done && residual_start;
    assign residual_stall = residual_input_valid && !residual_start;
    assign stall_pipeline = softmax_stall ||
                            rope_stall ||
                            rmsnorm_stall ||
                            residual_stall;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_data_reg         <= '0;
            s0_bias_reg         <= '0;
            s0_ctrl_reg         <= '0;
            s0_valid            <= 1'b0;
            s1_bias_reg         <= '0;
            s1_ctrl_reg         <= '0;
            s1_valid            <= 1'b0;
            s2_relu_reg         <= '0;
            s2_ctrl_reg         <= '0;
            s2_valid            <= 1'b0;
            s3_scale_reg        <= '0;
            s3_bypass_sel_reg   <= 1'b0;
            s3_valid            <= 1'b0;
            s4_handoff_reg      <= '0;
            s4_bypass_sel_reg   <= 1'b0;
            s4_valid            <= 1'b0;
            quant_inflight      <= 1'b0;
            rope_inflight       <= 1'b0;
            rope_result_valid   <= 1'b0;
            rope_result_reg     <= '0;
            softmax_result_valid <= 1'b0;
            softmax_result_reg  <= '0;
            softmax_result_mode_fp_reg <= 1'b0;
            residual_result_valid <= 1'b0;
            residual_result_reg <= '0;
            residual_active_mode_softmax <= 1'b0;
            residual_active_fp_mode <= 1'b0;
            residual_result_mode_softmax <= 1'b0;
            residual_result_fp_mode <= 1'b0;
            stream_out_reg      <= '0;
            stream_out_scale_reg <= '0;
            stream_out_valid_reg <= 1'b0;
        end else begin
            if (quant_issue)
                quant_inflight <= 1'b1;
            else if (quantize_valid)
                quant_inflight <= 1'b0;

            if (rope_start)
                rope_inflight <= 1'b1;
            else if (rope_done)
                rope_inflight <= 1'b0;

            if (rope_done) begin
                rope_result_reg <= rope_out;
                rope_result_valid <= 1'b1;
            end else if ((rmsnorm_start || residual_start) && rope_result_valid) begin
                rope_result_valid <= 1'b0;
            end

            if (chunked_softmax_out_valid && chunked_softmax_out_ready) begin
                softmax_result_reg <= chunked_softmax_out;
                softmax_result_mode_fp_reg <= chunked_softmax_out_mode_fp;
                softmax_result_valid <= 1'b1;
            end else if (softmax_result_accept) begin
                softmax_result_valid <= 1'b0;
            end

            if (residual_start) begin
                residual_active_mode_softmax <= (residual_op == RES_OP_PASS) &&
                                                !rope_en &&
                                                softmax_active_valid;
                residual_active_fp_mode <= (residual_op == RES_OP_PASS)
                                         ? (softmax_active_valid
                                            ? softmax_active_is_fp
                                            : (rope_en ? 1'b1 : fp_quant_mode))
                                         : 1'b1;
            end

            if (residual_row_valid) begin
                residual_result_reg <= residual_row_out;
                residual_result_valid <= 1'b1;
                residual_result_mode_softmax <= residual_active_mode_softmax;
                residual_result_fp_mode <= residual_active_fp_mode;
            end else if (quant_issue && residual_result_valid) begin
                residual_result_valid <= 1'b0;
            end

            if (quantize_valid) begin
                stream_out_reg       <= quantize_out;
                stream_out_scale_reg <= quantize_scale_out;
                stream_out_valid_reg <= 1'b1;
            end else if (stream_out_valid_reg && out_ready) begin
                stream_out_valid_reg <= 1'b0;
            end

            if (!stall_pipeline) begin
            // Stage 0: input capture
                s0_data_reg <= stream_in_data;
                s0_bias_reg <= stream_in_bias;
                s0_ctrl_reg <= vxm_ctrl;
                s0_valid    <= in_valid && !residual_emit_cmd;

                // Stage 1: bias-add capture
                s1_bias_reg <= s1_bias_next;
                s1_ctrl_reg <= s0_ctrl_reg[3:1];
                s1_valid    <= s0_valid;

                // Stage 2: ReLU capture
                s2_relu_reg <= s2_relu_next;
                s2_ctrl_reg <= s1_ctrl_reg[2:1];
                s2_valid    <= s1_valid;

                // Stage 3: scale/bypass capture
                s3_scale_reg      <= s3_scale_next;
                s3_bypass_sel_reg <= s2_ctrl_reg[1];
                s3_valid          <= s2_valid;

                // Stage 4: softmax/quantize handoff
                s4_handoff_reg    <= s3_scale_reg;
                s4_bypass_sel_reg <= s3_bypass_sel_reg;
                s4_valid          <= s3_valid;
            end
        end
    end

    softmax #(
        .LANES(LANES),
        .LANE_W(LANE_W),
        .MAX_CHUNKS(SOFTMAX_CHUNKS)
    ) softmax_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(chunked_softmax_in_valid),
        .x_in(s4_handoff_reg),
        .x_scale_i(8'sd0),
        .in_ready(chunked_softmax_in_ready),
        .out_valid(chunked_softmax_out_valid),
        .out_mode_fp(chunked_softmax_out_mode_fp),
        .y_out(chunked_softmax_out),
        .y_scale_o(chunked_softmax_out_scale),
        .out_ready(chunked_softmax_out_ready),
        .busy_o(chunked_softmax_busy)
    );

    vxm_rope #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) rope_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(rope_start),
        .x_in(mux_out),
        .cos_fp8(rope_cos_fp8),
        .sin_fp8(rope_sin_fp8),
        .y_out(rope_out),
        .done_o(rope_done),
        .busy_o(rope_busy)
    );

    assign pre_layernorm_in = rope_en ? rope_result_reg : mux_out;

    rmsnorm #(
        .LANES(LANES),
        .LANE_W(LANE_W),
        .CHUNKS(RMSNORM_CHUNKS)
    ) rmsnorm_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(rmsnorm_start),
        .x_in(pre_layernorm_in),
        .gamma(rmsnorm_gamma),
        .beta(rmsnorm_beta),
        .in_ready(rmsnorm_in_ready),
        .y_out(rmsnorm_out),
        .done_o(rmsnorm_done),
        .out_ready(rmsnorm_out_ready),
        .busy_o(rmsnorm_busy)
    );

    assign residual_row_in = residual_emit_cmd ? '0 :
                             (rmsnorm_bypass ? pre_layernorm_in : rmsnorm_out);

    residual_add #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) residual_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(residual_start),
        .op_i(residual_op),
        .row_i(residual_row_in),
        .ready_o(residual_ready),
        .busy_o(residual_busy),
        .done_o(residual_done),
        .row_valid_o(residual_row_valid),
        .row_o(residual_row_out),
        .acc_o(residual_acc_out)
    );

    quant #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) q_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(quant_issue),
        .quant_mode_i(quant_mode),
        .x_input(residual_result_reg),
        .out_valid(quantize_valid),
        .q_row_out(quantize_out),
        .q_scale_out(quantize_scale_out)
    );

    assign stream_out = stream_out_reg;
    assign stream_out_scale = stream_out_scale_reg;
    assign out_valid = stream_out_valid_reg;
    assign in_ready = !stall_pipeline;

endmodule
