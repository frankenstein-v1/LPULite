import lpu_pkg::*;

module sxm (
    input  logic clk,
    input  logic rst_n,
    
    // The 24-bit Control Word (12 bits for Input router, 12 bits for Weight router)
    // Every 3 bits controls a single output lane's multiplexer
    input  logic [11:0] opcode_input,
    input  logic [11:0] opcode_weight,
    
    // Incoming Streams
    input  superlane_t stream_in_west,  // Input moving East
    input  superlane_t stream_in_east,  // Weights moving West
    
    // Outgoing Streams directly into the MXM
    output superlane_t stream_out_to_mxm_left, 
    output superlane_t stream_out_to_mxm_top   
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
            input_d1 <= stream_in_west;
            input_d2 <= input_d1;
            input_d3 <= input_d2;
            
            // Weights shift one box to the right every clock cycle
            weight_d1 <= stream_in_east;
            weight_d2 <= weight_d1;
            weight_d3 <= weight_d2;
        end
    end

    // --- 2. THE MULTIPLEXERS (Crossbars & Selectors) ---
    // This happens instantly. No clock required.
    always_comb begin
        // Default assignment to prevent Verilog from creating unwanted latches
        stream_out_to_mxm_left = '0;
        stream_out_to_mxm_top  = '0;

        // Loop through all 4 output lanes
        for (int i = 0; i < 4; i++) begin
            
            // Slice out the specific 3-bit instruction for this exact lane
            logic [2:0] ctrl_input  = opcode_input[i*3 +: 3];
            logic [2:0] ctrl_weight = opcode_weight[i*3 +: 3];
            
            // --- EASTBOUND ROUTER (INPUT) ---
            case (ctrl_input)
                // Crossbar / Straight Pass / Broadcast
                3'b000: stream_out_to_mxm_left[i*8 +: 8] = stream_in_west[0*8 +: 8];
                3'b001: stream_out_to_mxm_left[i*8 +: 8] = stream_in_west[1*8 +: 8];
                3'b010: stream_out_to_mxm_left[i*8 +: 8] = stream_in_west[2*8 +: 8];
                3'b011: stream_out_to_mxm_left[i*8 +: 8] = stream_in_west[3*8 +: 8];
                // Staggering (Delays)
                3'b100: stream_out_to_mxm_left[i*8 +: 8] = input_d1[i*8 +: 8];
                3'b101: stream_out_to_mxm_left[i*8 +: 8] = input_d2[i*8 +: 8];
                3'b110: stream_out_to_mxm_left[i*8 +: 8] = input_d3[i*8 +: 8];
                // Bubbles
                3'b111: stream_out_to_mxm_left[i*8 +: 8] = 8'd0;
            endcase

            // --- WESTBOUND ROUTER (WEIGHTS) ---
            case (ctrl_weight)
                // Crossbar / Straight Pass / Broadcast
                3'b000: stream_out_to_mxm_top[i*8 +: 8] = stream_in_east[0*8 +: 8];
                3'b001: stream_out_to_mxm_top[i*8 +: 8] = stream_in_east[1*8 +: 8];
                3'b010: stream_out_to_mxm_top[i*8 +: 8] = stream_in_east[2*8 +: 8];
                3'b011: stream_out_to_mxm_top[i*8 +: 8] = stream_in_east[3*8 +: 8];
                // Staggering (Delays)
                3'b100: stream_out_to_mxm_top[i*8 +: 8] = weight_d1[i*8 +: 8];
                3'b101: stream_out_to_mxm_top[i*8 +: 8] = weight_d2[i*8 +: 8];
                3'b110: stream_out_to_mxm_top[i*8 +: 8] = weight_d3[i*8 +: 8];
                // Bubbles
                3'b111: stream_out_to_mxm_top[i*8 +: 8] = 8'd0;
            endcase
        end
    end

endmodule
