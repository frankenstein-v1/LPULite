`timescale 1ns/1ps

module mac (
    input  logic               clk,
    input  logic               rst,
    input  logic               en,
    input  logic signed [7:0]  activation_in,
    input  logic               weight_load,
    
    input  logic signed [7:0]  weight_value,
    
    output logic signed [19:0] product
);

logic signed [7:0] weight_reg;



always_ff @(posedge clk or posedge rst) begin 

    if(rst) begin
        weight_reg <= 8'sd0;
        product <= 20'sd0;
    end 

    else if (weight_load) begin
        weight_reg <= weight_value;
        product <= 20'sd0;
    end 

    else if (en) begin
        product <= weight_reg * activation_in;
    end 

    else begin 
        product <= 20'sd0;
    end 
end 

endmodule
