`timescale 1ns/1ps
import lpu_pkg::*;

module westbound_bus_fabric #(
    parameter int PAYLOAD_W = 32
) (
    // External producer selection.
    input  westbound_producer_e producer_sel,

    // Other producer payload+valid pairs.
    input  logic [PAYLOAD_W-1:0] sxm_payload,
    input  logic                 sxm_valid,

    input  logic [PAYLOAD_W-1:0] mem0_payload,
    input  logic                 mem0_valid,

    input  logic [PAYLOAD_W-1:0] vxm_payload,
    input  logic                 vxm_valid,

    input  logic [PAYLOAD_W-1:0] mem1_payload,
    input  logic                 mem1_valid,

    // Shared westbound bus.
    output logic [PAYLOAD_W-1:0] westbound_payload,
    output logic                 westbound_valid
);

    // Shared westbound mux: one selected producer per cycle.
    westbound_bus #(
        .PAYLOAD_W(PAYLOAD_W)
    ) u_westbound_bus (
        .producer_sel(producer_sel),
        .sxm_payload(sxm_payload),
        .sxm_valid(sxm_valid),
        .mem0_payload(mem0_payload),
        .mem0_valid(mem0_valid),
        .vxm_payload(vxm_payload),
        .vxm_valid(vxm_valid),
        .mem1_payload(mem1_payload),
        .mem1_valid(mem1_valid),
        .westbound_payload(westbound_payload),
        .westbound_valid(westbound_valid)
    );

endmodule
