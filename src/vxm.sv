`timescale 1ns/1ns

module vxm #(
    parameter int LANES   = 4,
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

    // Outputs
    output logic [31:0] stream_out,
    output logic                    out_valid,
    input  logic                    out_ready
);

    localparam int ROW_W = LANES * LANE_W;

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

    logic [ROW_W-1:0] s1_bias_next; //value that will be added into the bias add register next clk
    logic [ROW_W-1:0] s2_relu_next; //value that will be in ReLU next clk
    logic [ROW_W-1:0] s3_scale_next; //value in scale

    //outputs for quantize & softmax & muxes
    logic [31:0]              quantize_out;
    logic                     quantize_valid;
    logic [3:0][ROW_W-1:0]    softmax_out_vec;
    logic [3:0]               softmax_valid_vec;
    logic [3:0]               softmax_in_ready_vec;
    logic [3:0]               softmax_launch_vec;
    logic                     softmax_stall;
    logic                     stall_pipeline;
    logic [1:0]               launch_idx;
    logic [1:0]               collect_idx;
    logic [ROW_W-1:0]         softmax_result_reg [0:3];
    logic                     softmax_result_valid [0:3];

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
    end
    logic                     softmax_active_valid;
    logic [ROW_W-1:0]         softmax_active_data;
    logic [LANES*LANE_W-1:0]  mux_out;
    logic                     mux_valid;
    logic                     quant_mode_softmax;
    logic                     quant_issue;
    logic                     quant_inflight;
    logic                     quant_slot_available;
    logic [31:0]              stream_out_reg;
    logic                     stream_out_valid_reg;

    //extend from 1 lane to 4 lanes

    function automatic logic signed [ALU_W-1:0] widen_lane(
        input logic signed [LANE_W-1:0] lane_value
    );
        widen_lane = {{(ALU_W-LANE_W){lane_value[LANE_W-1]}}, lane_value};
    endfunction

    //data for each lane
    always_comb begin
        logic signed [LANE_W-1:0] data_lane;
        logic signed [LANE_W-1:0] bias_lane;
        logic signed [ALU_W-1:0]  bias_add_out;
        logic signed [ALU_W-1:0]  mux1_out;

        //logic for stage 1 bias next data 
        s1_bias_next = '0;
        for (int i = 0; i < LANES; i++) begin
            data_lane = s0_data_reg[i*LANE_W +: LANE_W];
            bias_lane = s0_bias_reg[i*LANE_W +: LANE_W];
            bias_add_out = widen_lane(data_lane) + widen_lane(bias_lane);
            mux1_out = s0_ctrl_reg[0] ? bias_add_out : widen_lane(data_lane);
            s1_bias_next[i*LANE_W +: LANE_W] = mux1_out[LANE_W-1:0];
        end
    end

    always_comb begin
        logic signed [LANE_W-1:0] bias_lane;
        logic signed [ALU_W-1:0]  relu_out;

        //relu buffering logic
        s2_relu_next = '0;
        for (int i = 0; i < LANES; i++) begin
            bias_lane = s1_bias_reg[i*LANE_W +: LANE_W];
            if (widen_lane(bias_lane) < 0)
                relu_out = '0;
            else
                relu_out = widen_lane(bias_lane);

            if (s1_ctrl_reg[0])
                s2_relu_next[i*LANE_W +: LANE_W] = relu_out[LANE_W-1:0];
            else
                s2_relu_next[i*LANE_W +: LANE_W] = bias_lane;
        end
    end

    //scale buffering logic 
    always_comb begin
        logic signed [LANE_W-1:0] relu_lane;
        logic signed [ALU_W-1:0]  scale_out;

        s3_scale_next = '0;
        for (int i = 0; i < LANES; i++) begin
            relu_lane = s2_relu_reg[i*LANE_W +: LANE_W];
            scale_out = widen_lane(relu_lane) >>> 1;

            if (s2_ctrl_reg[0])
                s3_scale_next[i*LANE_W +: LANE_W] = scale_out[LANE_W-1:0];
            else
                s3_scale_next[i*LANE_W +: LANE_W] = relu_lane;
        end
    end
    
    // logic to route the launch signal to the correct engine
    always_comb begin
        softmax_launch_vec = '0;
        if (s4_valid && s4_bypass_sel_reg && !stall_pipeline) begin
            case (launch_idx)
                2'd0: softmax_launch_vec[0] = 1'b1;
                2'd1: softmax_launch_vec[1] = 1'b1;
                2'd2: softmax_launch_vec[2] = 1'b1;
                2'd3: softmax_launch_vec[3] = 1'b1;
            endcase
        end
    end

    // logic to determine the current active softmax result (either from register or direct output)
    logic             active_result_valid;
    logic [ROW_W-1:0] active_result_data;

    always_comb begin
        active_result_valid = 1'b0;
        active_result_data  = '0;
        case (collect_idx)
            2'd0: begin
                active_result_valid = softmax_result_valid[0] || softmax_valid_vec[0];
                active_result_data  = softmax_result_valid[0] ? softmax_result_reg[0] : softmax_out_vec[0];
            end
            2'd1: begin
                active_result_valid = softmax_result_valid[1] || softmax_valid_vec[1];
                active_result_data  = softmax_result_valid[1] ? softmax_result_reg[1] : softmax_out_vec[1];
            end
            2'd2: begin
                active_result_valid = softmax_result_valid[2] || softmax_valid_vec[2];
                active_result_data  = softmax_result_valid[2] ? softmax_result_reg[2] : softmax_out_vec[2];
            end
            2'd3: begin
                active_result_valid = softmax_result_valid[3] || softmax_valid_vec[3];
                active_result_data  = softmax_result_valid[3] ? softmax_result_reg[3] : softmax_out_vec[3];
            end
        endcase
    end

    assign softmax_active_valid = active_result_valid;
    assign softmax_active_data  = active_result_data;

    assign mux_out   = softmax_active_valid ? softmax_active_data : s4_handoff_reg;
    assign mux_valid = softmax_active_valid || (s4_valid && !s4_bypass_sel_reg);
    assign quant_mode_softmax = softmax_active_valid;
    assign quant_slot_available = !quant_inflight && (!stream_out_valid_reg || out_ready);
    assign quant_issue = mux_valid && quant_slot_available;

    logic target_in_ready;
    always_comb begin
        target_in_ready = 1'b0;
        case (launch_idx)
            2'd0: target_in_ready = softmax_in_ready_vec[0];
            2'd1: target_in_ready = softmax_in_ready_vec[1];
            2'd2: target_in_ready = softmax_in_ready_vec[2];
            2'd3: target_in_ready = softmax_in_ready_vec[3];
        endcase
    end

    assign softmax_stall = s4_valid && s4_bypass_sel_reg && !target_in_ready;
    assign stall_pipeline = softmax_stall ||
                            (s4_valid && !s4_bypass_sel_reg && !quant_issue);

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
            quant_inflight      <= 1'b0;
            stream_out_reg      <= '0;
            stream_out_valid_reg <= 1'b0;
        end else begin
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
                end
            end
            if (softmax_valid_vec[1]) begin
                if (2'd1 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[1]   <= softmax_out_vec[1];
                    softmax_result_valid[1] <= 1'b1;
                end
            end
            if (softmax_valid_vec[2]) begin
                if (2'd2 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[2]   <= softmax_out_vec[2];
                    softmax_result_valid[2] <= 1'b1;
                end
            end
            if (softmax_valid_vec[3]) begin
                if (2'd3 == collect_idx && quant_issue && softmax_active_valid) begin
                    // Do not store, it goes straight to quant!
                end else begin
                    softmax_result_reg[3]   <= softmax_out_vec[3];
                    softmax_result_valid[3] <= 1'b1;
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
                stream_out_valid_reg <= 1'b1;
            end else if (stream_out_valid_reg && out_ready) begin
                stream_out_valid_reg <= 1'b0;
            end

            if (!stall_pipeline) begin
            // Stage 0: input capture
                s0_data_reg <= stream_in_data;
                s0_bias_reg <= stream_in_bias;
                s0_ctrl_reg <= vxm_ctrl;
                s0_valid    <= in_valid;

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
                .x_in(s4_handoff_reg),
                .in_ready(softmax_in_ready_vec[k]),
                .out_valid(softmax_valid_vec[k]),
                .y_out(softmax_out_vec[k])
            );
        end
    endgenerate

    quant #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) q_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(quant_issue),
        .mode_softmax(quant_mode_softmax),
        .x_input(mux_out),
        .out_valid(quantize_valid),
        .q_row_out(quantize_out)
    );

    assign stream_out = stream_out_reg;
    assign out_valid = stream_out_valid_reg;
    assign in_ready = !stall_pipeline;

endmodule
