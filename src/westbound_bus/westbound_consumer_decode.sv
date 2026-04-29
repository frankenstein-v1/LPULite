`timescale 1ns/1ps
import lpu_pkg::*;

module westbound_consumer_decode (

    input westbound_consumer_e consumer_sel,

    input logic westbound_valid,

    output logic mxm_west_en,
    output logic sxm_west_en,
    output logic mem0_west_en,
    output logic vxm_west_en
);


always_comb begin
    // Policy: at most one westbound consumer enable can be asserted per cycle.
    // If westbound_valid is low, all enables remain low.
    mxm_west_en = 1'b0;
    sxm_west_en = 1'b0;
    mem0_west_en = 1'b0;
    vxm_west_en = 1'b0;

    if (westbound_valid) begin 
        unique case (consumer_sel)
        WC_MXM: begin 
            mxm_west_en = 1'b1;
        end 
        
        WC_SXM: begin 
            sxm_west_en = 1'b1;
        end 

        WC_MEM0: begin 
            mem0_west_en = 1'b1;
        end 

        WC_VXM: begin 
            vxm_west_en = 1'b1;
        end 

        default: begin 
            mxm_west_en = 1'b0;
            sxm_west_en = 1'b0; 
            mem0_west_en = 1'b0;
            vxm_west_en = 1'b0;
        end 

        endcase 
    end 

end 

endmodule
