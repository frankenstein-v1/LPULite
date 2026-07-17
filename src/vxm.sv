`timescale 1ns/1ns

module vxm #(
    parameter int LANES   = 8,
    parameter int LANE_W  = 32, // Upgraded to 32 bits
    parameter int ALU_W   = 32 // Upgraded ALU width to 32 bits per user request
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

    // LayerNorm control and parameters
    input  logic                    layernorm_bypass,
    input  logic [LANES*LANE_W-1:0] layernorm_gamma,
    input  logic [LANES*LANE_W-1:0] layernorm_beta,

    // Outputs
    output logic [LANES*8-1:0] stream_out,
    output logic [31:0] stream_out_scale,
    output logic                    out_valid,
    input  logic                    out_ready
);

    localparam int ROW_W = LANES * LANE_W;
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;
    localparam logic [31:0] FP32_HALF = 32'h3f00_0000;
    localparam logic [31:0] FP32_ONE  = 32'h3f80_0000;

    //stage 0 -> input registers for each lane 
    logic [ROW_W-1:0] s0_data_reg; 
    logic [ROW_W-1:0] s0_bias_reg;
    logic [3:0]       s0_ctrl_reg;
    logic             s0_fp_softmax_reg;
    logic             s0_valid;
    
    //stage 1-> bias add registers for each lane
    logic [ROW_W-1:0] s1_bias_reg;
    logic [2:0]       s1_ctrl_reg;
    logic             s1_fp_softmax_reg;
    logic             s1_valid;

    //stage 2 -> ReLU register for each lane
    logic [ROW_W-1:0] s2_relu_reg;
    logic [1:0]       s2_ctrl_reg;
    logic             s2_fp_softmax_reg;
    logic             s2_valid;

    //stage 3-> scale register for each lane
    logic [ROW_W-1:0] s3_scale_reg;
    logic             s3_bypass_sel_reg;
    logic             s3_fp_softmax_reg;
    logic             s3_valid;

    //stage 4-> softmax/quantization register
    logic [ROW_W-1:0] s4_handoff_reg;
    logic             s4_bypass_sel_reg;
    logic             s4_fp_softmax_reg;
    logic             s4_valid;

    logic [ROW_W-1:0] s1_bias_next; // integer bias/bypass result
    logic [ROW_W-1:0] s2_relu_next; // ReLU/bypass result
    logic [ROW_W-1:0] s3_scale_next; // integer scale/bypass result
    logic [ROW_W-1:0] fp_bias_result_word;
    logic [ROW_W-1:0] fp_scale_result_word;
    logic [LANES-1:0] fp_bias_done_vec;
    logic [LANES-1:0] fp_scale_done_vec;
    logic             fp_bias_inflight;
    logic             fp_scale_inflight;
    logic             fp_bias_launch;
    logic             fp_scale_launch;
    logic             fp_bias_wait;
    logic             fp_scale_wait;
    logic             fp_bias_done_all;
    logic             fp_scale_done_all;

    //outputs for quantize & softmax & muxes
    logic [LANES*8-1:0]       quantize_out;
    logic [31:0]              quantize_scale_out;
    logic                     quantize_valid;
    logic [3:0][ROW_W-1:0]    softmax_out_vec;
    logic [3:0]               softmax_valid_vec;
    logic [3:0]               softmax_in_ready_vec;
    logic [3:0]               softmax_out_mode_fp_vec;
    logic [3:0]               softmax_launch_vec;
    logic                     softmax_stall;
    logic                     stall_pipeline;
    logic [1:0]               launch_idx;
    logic [1:0]               collect_idx;
    logic [ROW_W-1:0]         softmax_result_reg [0:3];
    logic                     softmax_result_valid [0:3];
    logic [3:0]               softmax_result_is_fp;

    initial begin
        launch_idx = 2'd0;
        collect_idx = 2'd0;
        softmax_result_reg[0] = '0;
        softmax_result_reg[1] = '0;
        softmax_result_reg[2] = '0;
        softmax_result_reg[3] = '0;
        softmax_result_valid[0] = 1'b0;
        softmax_result_valid[1] = 1'b0;
        softmax_result_valid[2] = 1'b0;
        softmax_result_valid[3] = 1'b0;
        softmax_result_is_fp = 4'b0000;
    end
    logic                     softmax_active_valid;
    logic [ROW_W-1:0]         softmax_active_data;
    logic                     softmax_active_is_fp;
    logic [LANES*LANE_W-1:0]  mux_out;
    logic                     mux_valid;
    logic                     quant_mode_softmax;
    logic                     quant_fp_mode;
    logic                     quant_issue;
    logic                     quant_inflight;
    logic                     quant_slot_available;
    logic                     fp_bias_stall;
    logic                     fp_scale_stall;
    logic [LANES*8-1:0]       stream_out_reg;
    logic [31:0]              stream_out_scale_reg;
    logic                     stream_out_valid_reg;

    //extend from 1 lane to 4 lanes

    function automatic logic signed [ALU_W-1:0] widen_lane(
        input logic signed [LANE_W-1:0] lane_value
    );
        widen_lane = {{(ALU_W-LANE_W){lane_value[LANE_W-1]}}, lane_value};
    endfunction

    function automatic logic [31:0] fp32_relu_bits(
        input logic [31:0] fp_bits
    );
        begin
            if ((fp_bits[30:0] == 31'd0) || !fp_bits[31])
                fp32_relu_bits = fp_bits;
            else
                fp32_relu_bits = FP32_ZERO;
        end
    endfunction

    // stage buffering logic using generate loop for all lanes
    // (Using generate loops and continuous assigns to avoid iverilog compilation issues)
    generate
        for (genvar i = 0; i < LANES; i++) begin : g_vxm_lanes
            // Stage 1: bias add logic
            logic signed [LANE_W-1:0] data_lane;
            logic signed [LANE_W-1:0] bias_lane_s0;
            logic signed [ALU_W-1:0]  bias_add_out;
            logic signed [ALU_W-1:0]  mux1_out;
            logic [31:0]              fp_bias_result_lane;
            logic                     fp_bias_done_lane;

            assign data_lane    = s0_data_reg[i*LANE_W +: LANE_W];
            assign bias_lane_s0 = s0_bias_reg[i*LANE_W +: LANE_W];
            assign bias_add_out = widen_lane(data_lane) + widen_lane(bias_lane_s0);
            assign mux1_out     = s0_ctrl_reg[0] ? bias_add_out : widen_lane(data_lane);
            assign s1_bias_next[i*LANE_W +: LANE_W] = mux1_out[LANE_W-1:0];
            assign fp_bias_result_word[i*LANE_W +: LANE_W] = fp_bias_result_lane;
            assign fp_bias_done_vec[i] = fp_bias_done_lane;

            cvfpu_fp32_fma u_fp_bias_add (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (fp_bias_launch),
                .multiplicand_i (data_lane),
                .multiplier_i   (FP32_ONE),
                .addend_i       (bias_lane_s0),
                .result_o       (fp_bias_result_lane),
                .done_o         (fp_bias_done_lane),
                .busy_o         (/* unused */)
            );

            // Stage 2: ReLU logic
            logic signed [LANE_W-1:0] bias_lane_s1;
            logic signed [ALU_W-1:0]  relu_out_int;

            assign bias_lane_s1 = s1_bias_reg[i*LANE_W +: LANE_W];
            assign relu_out_int = (widen_lane(bias_lane_s1) < 0) ? '0 : widen_lane(bias_lane_s1);
            assign s2_relu_next[i*LANE_W +: LANE_W] = s1_ctrl_reg[0]
                                                    ? (s1_fp_softmax_reg
                                                        ? fp32_relu_bits(bias_lane_s1)
                                                        : relu_out_int[LANE_W-1:0])
                                                    : bias_lane_s1;

            // Stage 3: scale logic
            logic signed [LANE_W-1:0] relu_lane;
            logic signed [ALU_W-1:0]  scale_out_int;
            logic [31:0]              fp_scale_result_lane;
            logic                     fp_scale_done_lane;

            assign relu_lane    = s2_relu_reg[i*LANE_W +: LANE_W];
            assign scale_out_int = widen_lane(relu_lane) >>> 1;
            assign s3_scale_next[i*LANE_W +: LANE_W] = s2_ctrl_reg[0] ? scale_out_int[LANE_W-1:0] : relu_lane;
            assign fp_scale_result_word[i*LANE_W +: LANE_W] = fp_scale_result_lane;
            assign fp_scale_done_vec[i] = fp_scale_done_lane;

            cvfpu_fp32_fma u_fp_scale_half (
                .clk_i          (clk),
                .rst_ni         (rst_n),
                .start_i        (fp_scale_launch),
                .multiplicand_i (relu_lane),
                .multiplier_i   (FP32_HALF),
                .addend_i       (FP32_ZERO),
                .result_o       (fp_scale_result_lane),
                .done_o         (fp_scale_done_lane),
                .busy_o         (/* unused */)
            );
        end
    endgenerate
    
    // logic to route the launch signal to the correct engine
    assign softmax_launch_vec[0] = (s4_valid && s4_bypass_sel_reg && !stall_pipeline && launch_idx == 2'd0);
    assign softmax_launch_vec[1] = (s4_valid && s4_bypass_sel_reg && !stall_pipeline && launch_idx == 2'd1);
    assign softmax_launch_vec[2] = (s4_valid && s4_bypass_sel_reg && !stall_pipeline && launch_idx == 2'd2);
    assign softmax_launch_vec[3] = (s4_valid && s4_bypass_sel_reg && !stall_pipeline && launch_idx == 2'd3);

    // logic to determine the current active softmax result (either from register or direct output)
    logic             active_result_valid;
    logic [ROW_W-1:0] active_result_data;
    logic             active_result_is_fp;

    assign active_result_valid = (collect_idx == 2'd0) ? (softmax_result_valid[0] || softmax_valid_vec[0]) :
                                 (collect_idx == 2'd1) ? (softmax_result_valid[1] || softmax_valid_vec[1]) :
                                 (collect_idx == 2'd2) ? (softmax_result_valid[2] || softmax_valid_vec[2]) :
                                 (softmax_result_valid[3] || softmax_valid_vec[3]);

    assign active_result_data  = (collect_idx == 2'd0) ? (softmax_result_valid[0] ? softmax_result_reg[0] : softmax_out_vec[0]) :
                                 (collect_idx == 2'd1) ? (softmax_result_valid[1] ? softmax_result_reg[1] : softmax_out_vec[1]) :
                                 (collect_idx == 2'd2) ? (softmax_result_valid[2] ? softmax_result_reg[2] : softmax_out_vec[2]) :
                                 (softmax_result_valid[3] ? softmax_result_reg[3] : softmax_out_vec[3]);

    assign active_result_is_fp = (collect_idx == 2'd0) ? (softmax_result_valid[0] ? softmax_result_is_fp[0] : softmax_out_mode_fp_vec[0]) :
                                 (collect_idx == 2'd1) ? (softmax_result_valid[1] ? softmax_result_is_fp[1] : softmax_out_mode_fp_vec[1]) :
                                 (collect_idx == 2'd2) ? (softmax_result_valid[2] ? softmax_result_is_fp[2] : softmax_out_mode_fp_vec[2]) :
                                 (softmax_result_valid[3] ? softmax_result_is_fp[3] : softmax_out_mode_fp_vec[3]);

    assign softmax_active_valid = active_result_valid;
    assign softmax_active_data  = active_result_data;
    assign softmax_active_is_fp = active_result_is_fp;

    assign mux_out   = softmax_active_valid ? softmax_active_data : s4_handoff_reg;
    assign mux_valid = softmax_active_valid || (s4_valid && !s4_bypass_sel_reg);
    assign quant_mode_softmax = softmax_active_valid;
    assign quant_fp_mode = softmax_active_valid ? softmax_active_is_fp : s4_fp_softmax_reg;
    assign quant_slot_available = !quant_inflight && (!stream_out_valid_reg || out_ready);
    assign quant_issue = mux_valid && quant_slot_available;
    assign fp_bias_wait = s0_valid && s0_fp_softmax_reg && s0_ctrl_reg[0];
    assign fp_scale_wait = s2_valid && s2_fp_softmax_reg && s2_ctrl_reg[0];
    assign fp_bias_launch = fp_bias_wait && !fp_bias_inflight;
    assign fp_scale_launch = fp_scale_wait && !fp_scale_inflight;
    assign fp_bias_done_all = &fp_bias_done_vec;
    assign fp_scale_done_all = &fp_scale_done_vec;
    assign fp_bias_stall = fp_bias_wait && !fp_bias_done_all;
    assign fp_scale_stall = fp_scale_wait && !fp_scale_done_all;

    logic target_in_ready;
    assign target_in_ready = (launch_idx == 2'd0) ? softmax_in_ready_vec[0] :
                             (launch_idx == 2'd1) ? softmax_in_ready_vec[1] :
                             (launch_idx == 2'd2) ? softmax_in_ready_vec[2] :
                             softmax_in_ready_vec[3];

    assign softmax_stall = s4_valid && s4_bypass_sel_reg && !target_in_ready;
    assign stall_pipeline = fp_bias_stall ||
                            fp_scale_stall ||
                            softmax_stall ||
                            (s4_valid && !s4_bypass_sel_reg && !quant_issue);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_data_reg         <= '0;
            s0_bias_reg         <= '0;
            s0_ctrl_reg         <= '0;
            s0_fp_softmax_reg   <= 1'b0;
            s0_valid            <= 1'b0;
            s1_bias_reg         <= '0;
            s1_ctrl_reg         <= '0;
            s1_fp_softmax_reg   <= 1'b0;
            s1_valid            <= 1'b0;
            s2_relu_reg         <= '0;
            s2_ctrl_reg         <= '0;
            s2_fp_softmax_reg   <= 1'b0;
            s2_valid            <= 1'b0;
            s3_scale_reg        <= '0;
            s3_bypass_sel_reg   <= 1'b0;
            s3_fp_softmax_reg   <= 1'b0;
            s3_valid            <= 1'b0;
            s4_handoff_reg      <= '0;
            s4_bypass_sel_reg   <= 1'b0;
            s4_fp_softmax_reg   <= 1'b0;
            s4_valid            <= 1'b0;
            launch_idx          <= 2'd0;
            collect_idx         <= 2'd0;
            softmax_result_reg[0]   <= '0;
            softmax_result_reg[1]   <= '0;
            softmax_result_reg[2]   <= '0;
            softmax_result_reg[3]   <= '0;
            softmax_result_valid[0] <= 1'b0;
            softmax_result_valid[1] <= 1'b0;
            softmax_result_valid[2] <= 1'b0;
            softmax_result_valid[3] <= 1'b0;
            softmax_result_is_fp    <= 4'b0000;
            fp_bias_inflight    <= 1'b0;
            fp_scale_inflight   <= 1'b0;
            quant_inflight      <= 1'b0;
            stream_out_reg      <= '0;
            stream_out_scale_reg <= '0;
            stream_out_valid_reg <= 1'b0;
        end else begin
            if (fp_bias_launch)
                fp_bias_inflight <= 1'b1;
            else if (fp_bias_inflight && fp_bias_done_all)
                fp_bias_inflight <= 1'b0;

            if (fp_scale_launch)
                fp_scale_inflight <= 1'b1;
            else if (fp_scale_inflight && fp_scale_done_all)
                fp_scale_inflight <= 1'b0;

            // Launch idx update
            if (s4_valid && s4_bypass_sel_reg && !stall_pipeline) begin
                launch_idx <= launch_idx + 2'd1;
            end

            // Capture newly completed softmax outputs into their registers (statically unrolled)
            if (softmax_valid_vec[0]) begin
                if (2'd0 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[0]   <= softmax_out_vec[0];
                    softmax_result_valid[0] <= 1'b1;
                    softmax_result_is_fp[0] <= softmax_out_mode_fp_vec[0];
                end
            end
            if (softmax_valid_vec[1]) begin
                if (2'd1 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[1]   <= softmax_out_vec[1];
                    softmax_result_valid[1] <= 1'b1;
                    softmax_result_is_fp[1] <= softmax_out_mode_fp_vec[1];
                end
            end
            if (softmax_valid_vec[2]) begin
                if (2'd2 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[2]   <= softmax_out_vec[2];
                    softmax_result_valid[2] <= 1'b1;
                    softmax_result_is_fp[2] <= softmax_out_mode_fp_vec[2];
                end
            end
            if (softmax_valid_vec[3]) begin
                if (2'd3 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[3]   <= softmax_out_vec[3];
                    softmax_result_valid[3] <= 1'b1;
                    softmax_result_is_fp[3] <= softmax_out_mode_fp_vec[3];
                end
            end

            // Handle the issue/collection of the current collect_idx
            if (quant_issue && softmax_active_valid) begin
                collect_idx <= collect_idx + 2'd1;
                case (collect_idx)
                    2'd0: softmax_result_valid[0] <= 1'b0;
                    2'd1: softmax_result_valid[1] <= 1'b0;
                    2'd2: softmax_result_valid[2] <= 1'b0;
                    2'd3: softmax_result_valid[3] <= 1'b0;
                endcase
            end

            if (quant_issue)
                quant_inflight <= 1'b1;
            else if (quantize_valid)
                quant_inflight <= 1'b0;

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
                s0_fp_softmax_reg <= fp_quant_mode;
                s0_valid    <= in_valid;

                // Stage 1: bias-add capture
                s1_bias_reg <= (s0_fp_softmax_reg && s0_ctrl_reg[0]) ? fp_bias_result_word : s1_bias_next;
                s1_ctrl_reg <= s0_ctrl_reg[3:1];
                s1_fp_softmax_reg <= s0_fp_softmax_reg;
                s1_valid    <= s0_valid;

                // Stage 2: ReLU capture
                s2_relu_reg <= s2_relu_next;
                s2_ctrl_reg <= s1_ctrl_reg[2:1];
                s2_fp_softmax_reg <= s1_fp_softmax_reg;
                s2_valid    <= s1_valid;

                // Stage 3: scale/bypass capture
                s3_scale_reg      <= (s2_fp_softmax_reg && s2_ctrl_reg[0]) ? fp_scale_result_word : s3_scale_next;
                s3_bypass_sel_reg <= s2_ctrl_reg[1];
                s3_fp_softmax_reg <= s2_fp_softmax_reg;
                s3_valid          <= s2_valid;

                // Stage 4: softmax/quantize handoff
                s4_handoff_reg    <= s3_scale_reg;
                s4_bypass_sel_reg <= s3_bypass_sel_reg;
                s4_fp_softmax_reg <= s3_fp_softmax_reg;
                s4_valid          <= s3_valid;
            end
        end
    end

    genvar k;
    generate
        for (k = 0; k < 4; k++) begin : gen_softmax
            softmax #(
                .LANES(LANES),
                .LANE_W(LANE_W)
            ) softmax_inst (
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(softmax_launch_vec[k]),
                .input_mode_fp(s4_fp_softmax_reg),
                .x_in(s4_handoff_reg),
                .in_ready(softmax_in_ready_vec[k]),
                .out_valid(softmax_valid_vec[k]),
                .out_mode_fp(softmax_out_mode_fp_vec[k]),
                .y_out(softmax_out_vec[k])
            );
        end
    endgenerate

    logic [LANES*LANE_W-1:0] layernorm_out;
    logic [LANES*LANE_W-1:0] layernorm_mux_out;

    lut_layernorm #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) layernorm_inst (
        .x_in(mux_out),
        .gamma(layernorm_gamma),
        .beta(layernorm_beta),
        .y_out(layernorm_out)
    );

    assign layernorm_mux_out = layernorm_bypass ? mux_out : layernorm_out;

    quant #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) q_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(quant_issue),
        .mode_softmax(quant_mode_softmax),
        .fp_quant_mode(quant_fp_mode),
        .softmax_input_is_fp(softmax_active_is_fp),
        .x_input(layernorm_mux_out),
        .out_valid(quantize_valid),
        .q_row_out(quantize_out),
        .q_scale_out(quantize_scale_out)
    );

    assign stream_out = stream_out_reg;
    assign stream_out_scale = stream_out_scale_reg;
    assign out_valid = stream_out_valid_reg;
    assign in_ready = !stall_pipeline;

endmodule
