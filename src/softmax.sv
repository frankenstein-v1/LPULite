// ==============================================================================
// Mathematical Core & Architecture Inspiration Credit: SuryaHead
// Repository: https://github.com/SurjaHead/softmax-in-hardware
// Completely structurally rewritten to support parallel LANES and AXI4-Stream 
// for the tinyLPU architecture.
// ==============================================================================
`default_nettype none
`timescale 1ns/1ps

module softmax #(
    parameter int LANES   = 4,
    parameter int LANE_W  = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     in_valid,
    input  logic [LANES*LANE_W-1:0]  x_in,

    output logic                     in_ready,
    output logic                     out_valid,
    output logic [LANES*LANE_W-1:0]  y_out
);

    localparam int MAX_BITS = 30;
    localparam int OUT_BITS = 8;
    localparam int SHIFT = MAX_BITS - OUT_BITS;

    // ---------------------------------------------------------
    // 1. Unpack Input Lanes
    // ---------------------------------------------------------
    logic signed [LANE_W-1:0] lane_data [0:LANES-1];
    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            lane_data[i] = x_in[i*LANE_W +: LANE_W];
        end
    end

    // ---------------------------------------------------------
    // 2. Combinational Max & Subtract
    // ---------------------------------------------------------
    logic signed [LANE_W-1:0] max_01, max_23, lane_max;
    assign max_01 = (lane_data[0] > lane_data[1]) ? lane_data[0] : lane_data[1];
    assign max_23 = (lane_data[2] > lane_data[3]) ? lane_data[2] : lane_data[3];
    assign lane_max = (max_01 > max_23) ? max_01 : max_23;

    logic signed [LANE_W-1:0] lane_sub [0:LANES-1];
    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            lane_sub[i] = lane_data[i] - lane_max;
        end
    end

    // ---------------------------------------------------------
    // 3. Parallel Exponentiation
    // ---------------------------------------------------------
    logic signed [LANE_W-1:0] lane_exp [0:LANES-1];
    genvar i;
    generate
        for (i = 0; i < LANES; i++) begin : gen_exp
            exp #(.DW(LANE_W)) exp_inst (
                .clk(clk),
                .rst(~rst_n),
                .q(lane_sub[i]),
                .q_out(lane_exp[i])
            );
        end
    endgenerate

    // ---------------------------------------------------------
    // 4. Combinational Sum
    // ---------------------------------------------------------
    logic signed [LANE_W-1:0] sum_exp;
    assign sum_exp = lane_exp[0] + lane_exp[1] + lane_exp[2] + lane_exp[3];

    // ---------------------------------------------------------
    // 5. State Machine & Registers
    // ---------------------------------------------------------
    typedef enum logic [1:0] { IDLE, DIVIDE, DONE_ST } state_t;
    state_t state, next_state;

    logic signed [LANE_W-1:0] lane_exp_reg [0:LANES-1];
    logic signed [LANE_W-1:0] sum_exp_reg;
    logic divider_start;
    logic divider_done;
    logic [LANE_W-1:0] quotient, remainder;

    assign in_ready = (state == IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            sum_exp_reg <= '0;
            for (int j = 0; j < LANES; j++) lane_exp_reg[j] <= '0;
        end else begin
            state <= next_state;
            if (state == IDLE && in_valid) begin
                sum_exp_reg <= sum_exp;
                for (int j = 0; j < LANES; j++) lane_exp_reg[j] <= lane_exp[j];
            end
        end
    end

    always_comb begin
        next_state = state;
        divider_start = 1'b0;
        out_valid = 1'b0;

        case (state)
            IDLE: begin
                if (in_valid) begin
                    divider_start = 1'b1;
                    next_state = DIVIDE;
                end
            end
            DIVIDE: begin
                if (divider_done) begin
                    next_state = DONE_ST;
                end
            end
            DONE_ST: begin
                out_valid = 1'b1;
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // ---------------------------------------------------------
    // 6. Reciprocal Divider
    // ---------------------------------------------------------
    divider #(.DW(LANE_W)) div_inst (
        .clk(clk),
        .rst(~rst_n),
        .start(divider_start),
        .dividend(32'd1073741824), // 1 << 30
        .divisor(sum_exp_reg),
        .quotient(quotient),
        .remainder(remainder),
        .done(divider_done)
    );

    // ---------------------------------------------------------
    // 7. Output Normalization
    // ---------------------------------------------------------
    logic signed [LANE_W-1:0] lane_out [0:LANES-1];
    always_comb begin
        for (int k = 0; k < LANES; k++) begin
            lane_out[k] = (quotient * lane_exp_reg[k]) >> SHIFT;
            y_out[k*LANE_W +: LANE_W] = lane_out[k];
        end
    end

endmodule
