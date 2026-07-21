`timescale 1ns/1ps
`include "lpu_pkg.sv"


module eastbound_bus #(
    // 8 signed int32 fixed-point lanes plus one shared row scale.
    parameter int PAYLOAD_E = $bits(eastbound_row_t)
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

    // Shared eastbound bus output: [255:0] = 8x int32 lanes, [263:256] = scale.
    output logic [PAYLOAD_E-1:0] eastbound_payload, 
    output logic                 eastbound_valid
);

    logic [4:0][PAYLOAD_E-1:0] producer_payloads;
    logic [4:0]                producer_valids;

    always_comb begin
        producer_payloads = '0;
        producer_valids   = '0;

        producer_payloads[EB_MXM]  = mxm_payload_e;
        producer_valids[EB_MXM]    = mxm_valid_e;
        producer_payloads[EB_SXM]  = sxm_payload_e;
        producer_valids[EB_SXM]    = sxm_valid_e;
        producer_payloads[EB_MEM0] = mem0_payload_e;
        producer_valids[EB_MEM0]   = mem0_valid_e;
        producer_payloads[EB_VXM]  = vxm_payload_e;
        producer_valids[EB_VXM]    = vxm_valid_e;
    end

    shared_bus_mux #(
        .PAYLOAD_W(PAYLOAD_E),
        .N_SOURCES(5),
        .SEL_W($bits(eastbound_producer_e))
    ) u_shared_bus_mux (
        .select_i(producer_sel),
        .payload_i(producer_payloads),
        .valid_i(producer_valids),
        .payload_o(eastbound_payload),
        .valid_o(eastbound_valid)
    );

endmodule
