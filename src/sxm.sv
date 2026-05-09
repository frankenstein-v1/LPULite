import lpu_pkg::*;

module sxm (
    input  logic clk,
    input  logic rst_n,
    
    // The 24-bit Control Word (12 bits for Input router, 12 bits for Weight router)
    // Every 3 bits controls a single output lane's multiplexer
    input  logic [11:0] opcode_input,
    input  logic [11:0] opcode_weight,
    
    // Eastbound Bus
    input  superlane_t eastbound_in,
    output superlane_t eastbound_out,
    
    // Westbound Bus
    input  superlane_t westbound_in,
    output superlane_t westbound_out
);

    // --- 1. THE DELAY LINES (FIFOs) ---
    // We physically build 3 layers of D flip-flops to hold the superlanes
    superlane_t input_d1, input_d2, input_d3;
    superlane_t weight_d1, weight_d2, weight_d3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_d1 <= '0; input_d2 <= '0; input_d3 <= '0;
            weight_d1 <= '0; weight_d2 <= '0; weight_d3 <= '0;
        end else begin
            // Input shifts one box to the right every clock cycle
            input_d1 <= eastbound_in;
            input_d2 <= input_d1;
            input_d3 <= input_d2;
            
            // Weights shift one box to the left every clock cycle
            weight_d1 <= westbound_in;
            weight_d2 <= weight_d1;
            weight_d3 <= weight_d2;
        end
    end

    // --- 2. THE MULTIPLEXERS (Crossbars & Selectors) ---
    // This happens instantly. No clock required.
    logic [7:0] router_eb_out [3:0];
    logic [7:0] router_wb_out [3:0];

    logic [7:0] eb_lane [3:0];
    logic [7:0] wb_lane [3:0];
    logic [7:0] id1_lane [3:0];
    logic [7:0] id2_lane [3:0];
    logic [7:0] id3_lane [3:0];
    logic [7:0] wd1_lane [3:0];
    logic [7:0] wd2_lane [3:0];
    logic [7:0] wd3_lane [3:0];

    generate
        for (genvar j = 0; j < 4; j++) begin : g_slice
            assign eb_lane[j] = eastbound_in[j*8 +: 8];
            assign wb_lane[j] = westbound_in[j*8 +: 8];
            assign id1_lane[j] = input_d1[j*8 +: 8];
            assign id2_lane[j] = input_d2[j*8 +: 8];
            assign id3_lane[j] = input_d3[j*8 +: 8];
            assign wd1_lane[j] = weight_d1[j*8 +: 8];
            assign wd2_lane[j] = weight_d2[j*8 +: 8];
            assign wd3_lane[j] = weight_d3[j*8 +: 8];
        end

        for (genvar i = 0; i < 4; i++) begin : g_router
            logic [2:0] cur_op_input;
            logic [2:0] cur_op_weight;
            
            assign cur_op_input  = opcode_input[i*3 +: 3];
            assign cur_op_weight = opcode_weight[i*3 +: 3];

            always_comb begin
                // Default assignments
                router_eb_out[i] = 8'd0;
                router_wb_out[i] = 8'd0;

                // --- EASTBOUND ROUTER (INPUT) ---
                case (cur_op_input)
                    // Crossbar / Straight Pass / Broadcast
                    3'b000: router_eb_out[i] = eb_lane[0];
                    3'b001: router_eb_out[i] = eb_lane[1];
                    3'b010: router_eb_out[i] = eb_lane[2];
                    3'b011: router_eb_out[i] = eb_lane[3];
                    // Staggering (Delays)
                    3'b100: router_eb_out[i] = id1_lane[i];
                    3'b101: router_eb_out[i] = id2_lane[i];
                    3'b110: router_eb_out[i] = id3_lane[i];
                    // Bubbles
                    3'b111: router_eb_out[i] = 8'd0;
                endcase

                // --- WESTBOUND ROUTER (WEIGHTS) ---
                case (cur_op_weight)
                    // Crossbar / Straight Pass / Broadcast
                    3'b000: router_wb_out[i] = wb_lane[0];
                    3'b001: router_wb_out[i] = wb_lane[1];
                    3'b010: router_wb_out[i] = wb_lane[2];
                    3'b011: router_wb_out[i] = wb_lane[3];
                    // Staggering (Delays)
                    3'b100: router_wb_out[i] = wd1_lane[i];
                    3'b101: router_wb_out[i] = wd2_lane[i];
                    3'b110: router_wb_out[i] = wd3_lane[i];
                    // Bubbles
                    3'b111: router_wb_out[i] = 8'd0;
                endcase
            end
            
            assign eastbound_out[i*8 +: 8] = router_eb_out[i];
            assign westbound_out[i*8 +: 8] = router_wb_out[i];
        end
    endgenerate

endmodule
