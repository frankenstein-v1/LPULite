module vxm #(
    parameter int LANES   = 4,
    parameter int LANE_W  = 8,
    parameter int ALU_W   = 16,
    parameter int ACCUM_W = 20
) (
    input  logic clk,
    input  logic rst_n,

    // Input vectors, one signed lane per chunk of LANE_W bits.
    input  logic [LANES*LANE_W-1:0] stream_in_data,
    input  logic [LANES*LANE_W-1:0] stream_in_param,
    input  logic                    in_valid,
    output logic                    in_ready,

    // 00 = ReLU, 01 = Add, 10 = Multiply, 11 = Pass-through
    input  logic [1:0] math_op,

    // Control the stateful accumulator independently from output emission.
    input  logic accum_en,
    input  logic emit_result,
    input  logic flush,
    input  logic clear_accum,

    output logic [LANES*ACCUM_W-1:0] stream_out,
    output logic                     out_valid,
    input  logic                     out_ready
);

    localparam logic [1:0] VXM_OP_RELU = 2'b00;
    localparam logic [1:0] VXM_OP_ADD  = 2'b01;
    localparam logic [1:0] VXM_OP_MUL  = 2'b10;
    localparam logic [1:0] VXM_OP_PASS = 2'b11;

    logic signed [LANE_W-1:0]  data_lane     [0:LANES-1];
    logic signed [LANE_W-1:0]  param_lane    [0:LANES-1];
    logic signed [ALU_W-1:0]   alu_result    [0:LANES-1];
    logic signed [ACCUM_W-1:0] accum_reg     [0:LANES-1];

    logic [LANES*ACCUM_W-1:0] packed_alu_result;
    logic [LANES*ACCUM_W-1:0] packed_accum_result;
    logic [LANES*ACCUM_W-1:0] out_payload_reg;

    logic out_slot_available;
    logic take_input;
    logic do_emit_result;
    logic do_flush;

    function automatic logic signed [ALU_W-1:0] widen_lane(
        input logic signed [LANE_W-1:0] lane_value
    );
        widen_lane = {{(ALU_W-LANE_W){lane_value[LANE_W-1]}}, lane_value};
    endfunction

    assign out_slot_available = !out_valid || out_ready;
    assign in_ready           = !emit_result || out_slot_available;
    assign take_input         = in_valid && in_ready;
    assign do_emit_result     = take_input && emit_result && !flush;
    assign do_flush           = flush && out_slot_available;

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            data_lane[i]  = stream_in_data[i*LANE_W +: LANE_W];
            param_lane[i] = stream_in_param[i*LANE_W +: LANE_W];

            unique case (math_op)
                VXM_OP_RELU: begin
                    if (data_lane[i] < 0)
                        alu_result[i] = '0;
                    else
                        alu_result[i] = widen_lane(data_lane[i]);
                end
                VXM_OP_ADD: begin
                    alu_result[i] = widen_lane(data_lane[i]) + widen_lane(param_lane[i]);
                end
                VXM_OP_MUL: begin
                    alu_result[i] = $signed(data_lane[i]) * $signed(param_lane[i]);
                end
                VXM_OP_PASS: begin
                    alu_result[i] = widen_lane(data_lane[i]);
                end
                default: begin
                    alu_result[i] = '0;
                end
            endcase

            packed_alu_result[i*ACCUM_W +: ACCUM_W]   = {{(ACCUM_W-ALU_W){alu_result[i][ALU_W-1]}}, alu_result[i]};
            packed_accum_result[i*ACCUM_W +: ACCUM_W] = accum_reg[i];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid        <= 1'b0;
            out_payload_reg  <= '0;
            for (int i = 0; i < LANES; i++) begin
                accum_reg[i] <= '0;
            end
        end else begin
            if (do_flush) begin
                out_payload_reg <= packed_accum_result;
                out_valid       <= 1'b1;
            end else if (do_emit_result) begin
                out_payload_reg <= packed_alu_result;
                out_valid       <= 1'b1;
            end else if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (clear_accum) begin
                for (int i = 0; i < LANES; i++) begin
                    accum_reg[i] <= '0;
                end
            end else if (take_input && accum_en) begin
                for (int i = 0; i < LANES; i++) begin
                    accum_reg[i] <= accum_reg[i] + {{(ACCUM_W-ALU_W){alu_result[i][ALU_W-1]}}, alu_result[i]};
                end
            end
        end
    end

    always_comb begin
        assert (ALU_W >= (LANE_W + 1))
            else $error("VXM ALU_W must be large enough for signed addition");
        assert (ALU_W >= (2 * LANE_W))
            else $error("VXM ALU_W must be large enough for signed multiply");
        assert (ACCUM_W >= ALU_W)
            else $error("VXM ACCUM_W must be at least ALU_W");
        assert (!(emit_result && flush))
            else $error("VXM emit_result and flush cannot be asserted together");
    end

    assign stream_out = out_payload_reg;

endmodule
