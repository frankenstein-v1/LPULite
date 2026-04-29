`timescale 1ns/1ps

module acc (
    input logic clk, 
    input logic rst, 
    input logic en, 
    input logic clear, 
    input logic signed [19:0] product_in, 
    output logic signed [31:0] sum_out
);

always_ff @(posedge clk or posedge rst) begin

    if(rst) begin 
        sum_out <= 32'sd0;
    end 

    else if(clear) begin 
        sum_out <= 32'sd0;
    end 

    else if(en) begin 
        sum_out <= sum_out + {{12{product_in[19]}}, product_in};
    end 


end 

endmodule
