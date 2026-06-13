`timescale 1ns/1ps
`include "lpu_pkg.sv"

module westbound_bus #(
    parameter int PAYLOAD_W = 32
) (
    // Producer select (encoded): choose exactly one source for this cycle.
    input  westbound_producer_e producer_sel,

    // SXM producer
    input  logic [PAYLOAD_W-1:0] sxm_payload,
    input  logic                 sxm_valid,

    // MEM0 producer
    input  logic [PAYLOAD_W-1:0] mem0_payload,
    input  logic                 mem0_valid,

    // VXM producer
    input  logic [PAYLOAD_W-1:0] vxm_payload,
    input  logic                 vxm_valid,

    // MEM1 producer
    input  logic [PAYLOAD_W-1:0] mem1_payload,
    input  logic                 mem1_valid,

    // Shared westbound bus output
    output logic [PAYLOAD_W-1:0] westbound_payload,
    output logic                 westbound_valid
);

    always_comb begin
        westbound_payload = '0;
        westbound_valid   = 1'b0;

        unique case (producer_sel)
            WB_SXM: begin
                westbound_payload = sxm_payload;
                westbound_valid   = sxm_valid;
            end

            WB_MEM0: begin
                westbound_payload = mem0_payload;
                westbound_valid   = mem0_valid;
            end

            WB_VXM: begin
                westbound_payload = vxm_payload;
                westbound_valid   = vxm_valid;
            end

            WB_MEM1: begin
                westbound_payload = mem1_payload;
                westbound_valid   = mem1_valid;
            end

            default: begin
                westbound_payload = '0;
                westbound_valid   = 1'b0;
            end
        endcase
    end

endmodule
