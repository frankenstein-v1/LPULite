`timescale 1ns/1ps
import lpu_pkg::*;


module eastbound_bus #(
    parameter int PAYLOAD_E = 32
)(

    input eastbound_producer_e producer_sel,

    //mxm producer 
    input  logic [PAYLOAD_E-1:0] mxm_payload_e,
    input logic                  mxm_valid_e, 

    //vxm producer 

    input logic [PAYLOAD_E-1:0] vxm_payload_e, 
    input logic                 vxm_valid_e,

    //sxm prod
    input logic [PAYLOAD_E-1:0] sxm_payload_e, 
    input logic                 sxm_valid_e, 

    //mem0 prod 
    input logic [PAYLOAD_E-1:0] mem0_payload_e, 
    input logic                 mem0_valid_e, 

    output logic [PAYLOAD_E-1:0] eastbound_payload, 
    output logic                 eastbound_valid
);

    always_comb begin
        eastbound_payload = '0;
        eastbound_valid = 1'b0;

        unique case(producer_sel)
        EB_MXM: begin
            eastbound_payload = mxm_payload_e;
            eastbound_valid = mxm_valid_e;
        end 

        EB_VXM: begin 
            eastbound_payload = vxm_payload_e;
            eastbound_valid = vxm_valid_e;
        end 

        EB_SXM: begin 
            eastbound_payload = sxm_payload_e;
            eastbound_valid = sxm_valid_e;
        end 

        EB_MEM0: begin 
            eastbound_payload = mem0_payload_e;
            eastbound_valid = mem0_valid_e;
        end 

    endcase
    end 

endmodule
