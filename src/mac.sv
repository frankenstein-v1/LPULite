`timescale 1ns/1ps

module mac (
    input  logic               clk,
    input  logic               rst,
    input  logic               en,
    input  logic        [7:0]  input_in,
    input  logic               input_is_signed,
    input  logic               weight_load,
    
    input  logic        [7:0]  weight_value,
    input  logic               weight_is_signed,
    
    output logic signed [19:0] product
);

logic [7:0]        weight_reg;
logic              weight_is_signed_reg;
logic signed [8:0] input_ext;
logic signed [8:0] weight_ext;

// Sign-extend input and weight operands combinationally
// (Implemented as continuous assigns for maximum iverilog compatibility)
assign input_ext = input_is_signed
    ? $signed({input_in[7], input_in})
    : $signed({1'b0, input_in});

assign weight_ext = weight_is_signed_reg
    ? $signed({weight_reg[7], weight_reg})
    : $signed({1'b0, weight_reg});



always_ff @(posedge clk or posedge rst) begin 

    if(rst) begin
        weight_reg <= 8'd0;
        weight_is_signed_reg <= 1'b1;
        product <= 20'sd0;
    end 

    else if (weight_load) begin
        weight_reg <= weight_value;
        weight_is_signed_reg <= weight_is_signed;
        product <= 20'sd0;
    end 

    else if (en) begin
        product <= weight_ext * input_ext;
    end 

    else begin 
        product <= 20'sd0;
    end 
end 

endmodule
