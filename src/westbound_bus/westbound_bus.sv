`timescale 1ns/1ps
`include "lpu_pkg.sv"

module westbound_bus (
    // Producer select (encoded): choose exactly one source for this cycle.
    input  westbound_producer_e producer_sel,

    // SXM producer
    input  westbound_row_t       sxm_payload,
    input  logic                 sxm_valid,

    // MEM0 producer
    input  westbound_row_t       mem0_payload,
    input  logic                 mem0_valid,

    // VXM producer
    input  westbound_row_t       vxm_payload,
    input  logic                 vxm_valid,

    // MEM1 producer
    input  westbound_row_t       mem1_payload,
    input  logic                 mem1_valid,

    // Shared westbound bus output: [63:0] = 8x int8 lanes, [71:64] = scale.
    output westbound_row_t       westbound_payload,
    output logic                 westbound_valid
);
    localparam int PAYLOAD_W = 72;

    logic [4:0][PAYLOAD_W-1:0] producer_payloads;
    logic [4:0]                 producer_valids;

    always_comb begin
        producer_payloads = '0;
        producer_valids   = '0;

        producer_payloads[WB_SXM]  = sxm_payload;
        producer_valids[WB_SXM]    = sxm_valid;
        producer_payloads[WB_MEM0] = mem0_payload;
        producer_valids[WB_MEM0]   = mem0_valid;
        producer_payloads[WB_VXM]  = vxm_payload;
        producer_valids[WB_VXM]    = vxm_valid;
        producer_payloads[WB_MEM1] = mem1_payload;
        producer_valids[WB_MEM1]   = mem1_valid;
    end

    shared_bus_mux #(
        .PAYLOAD_W(PAYLOAD_W),
        .N_SOURCES(5),
        .SEL_W(3)
    ) u_shared_bus_mux (
        .select_i(producer_sel),
        .payload_i(producer_payloads),
        .valid_i(producer_valids),
        .payload_o(westbound_payload),
        .valid_o(westbound_valid)
    );

endmodule
