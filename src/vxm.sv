import lpu_pkg::*;

module vxm (
    input  logic clk,
    input  logic rst_n,
    
    // Control
    input  logic [1:0] opcode, // 00:ReLU, 01:Bias+, 10:Scale, 11:Accumulate
    
    // Incoming Streams
    input  superlane_t stream_in_data,
    input  superlane_t stream_in_param,
    
    // Outgoing Stream
    output superlane_t stream_out
);

    // Internal wire to hold the raw math results before the flip-flop catches them
    superlane_t alu_result;

    // Combinational Logic (The Math & Muxes)
    // always_comb means this is pure physical wire and gates, no clock required
    always_comb begin
        // Loop through the 4 lanes. 
        for (int i = 0; i < 4; i++) begin
            
            // Extract the 8-bit lane from the 32-bit superlanes.
            // The 'signed' keyword is critical here. It tells the synthesizer 
            // to treat the top bit as a sign bit (positive/negative) instead of 
            // just a really big number.
            logic signed [7:0] data_lane;
            logic signed [7:0] param_lane;
            
            data_lane  = stream_in_data[i*8 +: 8];
            param_lane = stream_in_param[i*8 +: 8];
            
            // The Multiplexer
            case (opcode)
                2'b00: begin // ReLU
                    // Because we declared data_lane as 'signed', this < 0 check actually works.
                    if (data_lane < 0) 
                        alu_result[i*8 +: 8] = 8'd0;
                    else 
                        alu_result[i*8 +: 8] = data_lane;
                end
                
                2'b01: begin // Bias+
                    alu_result[i*8 +: 8] = data_lane + param_lane;
                end
                
                2'b10: begin // Scale
                    // Multiplies the two 8-bit numbers. Since the lane is only 8 bits wide,
                    // the top 8 bits of the 16-bit result are naturally truncated.
                    alu_result[i*8 +: 8] = data_lane * param_lane;
                end
                
                2'b11: begin // Accumulate
                    // Physically identical to Bias+ since we have no internal memory
                    alu_result[i*8 +: 8] = data_lane + param_lane;
                end
                
                default: alu_result[i*8 +: 8] = 8'd0;
            endcase
        end
    end

    // Sequential Logic (The Conveyor Belt)
    // This catches the alu_result and pushes it out on the positive clock edge
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stream_out <= '0;
        end else begin
            stream_out <= alu_result;
        end
    end

endmodule