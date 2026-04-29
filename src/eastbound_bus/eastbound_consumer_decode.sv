`timescale 1ns/1ps
import lpu_pkg::*;

module eastbound_consumer_decode(
    input eastbound_consumer_e consumer_sel,

    input logic eastbound_valid, 

    output logic sxm_east_en, 
    output logic mem0_east_en, 
    output logic vxm_east_en, 
    output logic mem1_east_en

);

always_comb begin 
    // Policy: at most one eastbound consumer enable can be asserted per cycle.
    // If eastbound_valid is low, all enables remain low.

    sxm_east_en = 1'b0;
    mem0_east_en = 1'b0;
    vxm_east_en = 1'b0;
    mem1_east_en =1'b0;

    if(eastbound_valid) begin 
        unique case (consumer_sel)
        EC_SXM: begin 
            sxm_east_en = 1'b1;
        end 

        EC_MEM0: begin 
            mem0_east_en = 1'b1;
        end 

        EC_VXM: begin 
            vxm_east_en = 1'b1;
        end

        EC_MEM1: begin 
            mem1_east_en = 1'b1;
        end

        default: begin 
            sxm_east_en = 1'b0;
            mem0_east_en = 1'b0;
            vxm_east_en = 1'b0;
            mem1_east_en = 1'b0;
        end 
        endcase 
    end 



end 

endmodule
