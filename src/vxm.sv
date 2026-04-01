import lpu_pkg::*;

module vxm (
    input  logic clk,
    input  logic rst_n,
    
    // The 32-bit Superlanes
    input  logic [31:0] stream_in_data,
    input  logic [31:0] stream_in_param,

    // The Compiler's Control Signals
    input  logic [1:0]  math_op,   // 00 = Pass, 01 = Add, 10 = Multiply
    input  logic        accum_en,  // 1 = Save to register, 0 = Ignore
    input  logic        flush,     // 1 = Dump register to output, 0 = Hold

    // The Output Superlane
    output logic [31:0] stream_out
);

    // 1. THE UNPACKING (Fixing the Lane Collision)
    // We create distinct 8-bit wires so the math can't bleed across lanes
    logic signed [7:0] data_lane  [0:3];
    logic signed [7:0] param_lane [0:3];
    logic signed [7:0] alu_result [0:3];
    logic signed [7:0] accum_reg  [0:3]; // Our new stateful registers

    always_comb begin
        // Manually slicing the 32-bit highway into 4 isolated lanes
        data_lane[0] = stream_in_data[7:0];
        data_lane[1] = stream_in_data[15:8];
        data_lane[2] = stream_in_data[23:16];
        data_lane[3] = stream_in_data[31:24];

        param_lane[0] = stream_in_param[7:0];
        param_lane[1] = stream_in_param[15:8];
        param_lane[2] = stream_in_param[23:16];
        param_lane[3] = stream_in_param[31:24];

        // 2. THE MATH (Safely isolated in 4 ALUs)
        for (int i = 0; i < 4; i++) begin
            case (math_op)
                2'b01: alu_result[i] = data_lane[i] + param_lane[i]; // Bias
                2'b10: alu_result[i] = data_lane[i] * param_lane[i]; // Scale
                default: alu_result[i] = data_lane[i];               // Pass-through
            endcase
        end
    end

    // 3. THE ACCUMULATOR (Our new inference optimization)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) accum_reg[i] <= 8'd0;
        end else if (flush) begin
            // Reset the registers after we dump the answer
            for (int i = 0; i < 4; i++) accum_reg[i] <= 8'd0;
        end else if (accum_en) begin
            // Keep a running total
            for (int i = 0; i < 4; i++) accum_reg[i] <= accum_reg[i] + alu_result[i];
        end
    end

    // 4. THE PACKING (Putting it back on the highway)
    // We use the {} concatenation operator to stitch the 4 lanes back into 32 bits.
    // We only push data when the compiler yells "flush", otherwise we blow zeros (bubbles).
    assign stream_out = flush ? {accum_reg[3], accum_reg[2], accum_reg[1], accum_reg[0]} : 32'd0;

endmodule