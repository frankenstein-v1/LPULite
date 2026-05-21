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
    logic [LANES*LANE_W-1:0]  softmax_out;
    logic                     softmax_valid;
    logic                     softmax_in_ready;
    logic                     softmax_hold;
    logic                     softmax_launch;
    logic                     stall_pipeline;
    logic [ROW_W-1:0]         softmax_result_reg;
    logic                     softmax_result_valid;
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
    
    //logic to turn registers representing state "n+1" into state "n"
    assign softmax_launch = s4_valid && s4_bypass_sel_reg && !softmax_hold &&
                            !softmax_result_valid && softmax_in_ready;

    assign mux_out = softmax_result_valid ? softmax_result_reg : s4_handoff_reg;
    assign mux_valid = softmax_result_valid || (s4_valid && !s4_bypass_sel_reg);
    assign quant_mode_softmax = softmax_result_valid;
    assign quant_slot_available = !quant_inflight && (!stream_out_valid_reg || out_ready);
    assign quant_issue = mux_valid && quant_slot_available;

    assign stall_pipeline = softmax_launch ||
                            (softmax_hold && !softmax_valid) ||
                            (mux_valid && !quant_issue);

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
            softmax_hold        <= 1'b0;
            softmax_result_reg  <= '0;
            softmax_result_valid <= 1'b0;
            quant_inflight      <= 1'b0;
            stream_out_reg      <= '0;
            stream_out_valid_reg <= 1'b0;
        end else begin
            if (softmax_launch)
                softmax_hold <= 1'b1;
            else if (softmax_valid)
                softmax_hold <= 1'b0;

            if (softmax_valid) begin
                softmax_result_reg   <= softmax_out;
                softmax_result_valid <= 1'b1;
            end else if (quant_issue && softmax_result_valid) begin
                softmax_result_valid <= 1'b0;
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

    softmax #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) softmax_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(softmax_launch),
        .x_in(s4_handoff_reg),
        .in_ready(softmax_in_ready),
        .out_valid(softmax_valid),
        .y_out(softmax_out)
    );

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
