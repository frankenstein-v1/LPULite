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
    input  logic [2:0]              vxm_ctrl,

    // Outputs
    output logic [LANES*LANE_W-1:0] stream_out,
    output logic                    out_valid,
    input  logic                    out_ready
);

    logic signed [LANE_W-1:0] data_lane [0:LANES-1];
    logic signed [LANE_W-1:0] bias_lane [0:LANES-1];

    logic signed [ALU_W-1:0]  bias_add_out [0:LANES-1];
    logic signed [ALU_W-1:0]  mux1_out     [0:LANES-1];
    logic signed [ALU_W-1:0]  relu_out     [0:LANES-1];
    logic signed [ALU_W-1:0]  mux2_out     [0:LANES-1];
    logic signed [ALU_W-1:0]  scale_out    [0:LANES-1];
    logic signed [ALU_W-1:0]  mux3_out     [0:LANES-1];
    
    logic [LANES*LANE_W-1:0]  reg_out;

    logic mux1_sel;
    logic mux2_sel;
    logic mux3_sel;

    assign mux1_sel = vxm_ctrl[0];
    assign mux2_sel = vxm_ctrl[1];
    assign mux3_sel = vxm_ctrl[2];

    function automatic logic signed [ALU_W-1:0] widen_lane(
        input logic signed [LANE_W-1:0] lane_value
    );
        widen_lane = {{(ALU_W-LANE_W){lane_value[LANE_W-1]}}, lane_value};
    endfunction

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            data_lane[i] = stream_in_data[i*LANE_W +: LANE_W];
            bias_lane[i] = stream_in_bias[i*LANE_W +: LANE_W];

            // 1. Bias Add
            bias_add_out[i] = widen_lane(data_lane[i]) + widen_lane(bias_lane[i]);

            // 2. Mux 1 (2-to-1): Input vs Bias Add
            mux1_out[i] = (mux1_sel) ? bias_add_out[i] : widen_lane(data_lane[i]);

            // 3. ReLU (feeds from Mux 1)
            if (mux1_out[i] < 0) begin
                relu_out[i] = '0;
            end else begin
                relu_out[i] = mux1_out[i];
            end

            // 4. Mux 2 (2-to-1): Mux 1 vs ReLU
            mux2_out[i] = (mux2_sel) ? relu_out[i] : mux1_out[i];

            // 5. Scale Function
            // Arithmetic right shift preserves the signed value while scaling by 2.
            // input x 1/sqrt(dk). 
            //in this case dk = 4, sqrt(dk) = 2, so multiply by 1/2 is the same as shifting right by 1
            scale_out[i] = mux2_out[i] >>> 1;

            // 6. Mux 3 (2-to-1): Mux 2 vs Scale
            mux3_out[i] = (mux3_sel) ? scale_out[i] : mux2_out[i];
            
        end
    end

   //collect the outputs as rows 
    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            reg_out[i*LANE_W +: LANE_W] = mux3_out[i];
        end
    end

    // Output assignment
    assign stream_out = reg_out;
    assign out_valid = in_valid;
    assign in_ready = out_ready;

endmodule
