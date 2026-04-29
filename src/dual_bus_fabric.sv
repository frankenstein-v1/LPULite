`timescale 1ns/1ps
import lpu_pkg::*;

module dual_bus_fabric #(
    parameter int PAYLOAD_W = 32,
    parameter int MXM_SIZE  = 4
) (
    // Clock/reset for consumer-side ingress capture registers.
    input  logic clk,
    input  logic rst_n,

    // selection inputs, who is allowed to input data on each bus 
    input  westbound_producer_e westbound_sel,
    input  eastbound_producer_e eastbound_sel,
    // Consumer-side selection is independent per direction.
    input  westbound_consumer_e westbound_consumer_sel,
    input  eastbound_consumer_e eastbound_consumer_sel,

    // MXM produces a matrix, so eastbound exports one selected cell per cycle.
    input  logic signed [PAYLOAD_W-1:0] mxm_out [MXM_SIZE-1:0][MXM_SIZE-1:0],
    input  logic [$clog2(MXM_SIZE)-1:0] mxm_e_row_sel,
    input  logic [$clog2(MXM_SIZE)-1:0] mxm_e_col_sel,
    input  logic                        mxm_e_valid_in,

    // SXM producer payload+valid for each direction.
    input  logic [PAYLOAD_W-1:0] sxm_payload_w,
    input  logic                 sxm_valid_w,
    input  logic [PAYLOAD_W-1:0] sxm_payload_e,
    input  logic                 sxm_valid_e,

    // VXM producer payload+valid for each direction.
    input  logic [PAYLOAD_W-1:0] vxm_payload_w,
    input  logic                 vxm_valid_w,
    input  logic [PAYLOAD_W-1:0] vxm_payload_e,
    input  logic                 vxm_valid_e,

    // MEM0 producer payload+valid for each direction.
    input  logic [PAYLOAD_W-1:0] mem0_payload_w,
    input  logic                 mem0_valid_w,
    input  logic [PAYLOAD_W-1:0] mem0_payload_e,
    input  logic                 mem0_valid_e,

    // MEM1 producer payload+valid for westbound only.
    input  logic [PAYLOAD_W-1:0] mem1_payload_w,
    input  logic                 mem1_valid_w,

    //the actual shared busses, the value payload of the bus and the validity of the data for W + E
    output logic [PAYLOAD_W-1:0] westbound_payload,
    output logic                 westbound_valid,
    output logic [PAYLOAD_W-1:0] eastbound_payload,
    output logic                 eastbound_valid,

    // Broadcast taps to each unit, this is done so the mxm can observe both the W + E for example, done for all components

    output logic [PAYLOAD_W-1:0] mxm_westbound_in,
    output logic                 mxm_westbound_valid_in,
    output logic [PAYLOAD_W-1:0] mxm_eastbound_in,
    output logic                 mxm_eastbound_valid_in,

    output logic [PAYLOAD_W-1:0] sxm_westbound_in,
    output logic                 sxm_westbound_valid_in,
    output logic [PAYLOAD_W-1:0] sxm_eastbound_in,
    output logic                 sxm_eastbound_valid_in,

    output logic [PAYLOAD_W-1:0] vxm_westbound_in,
    output logic                 vxm_westbound_valid_in,
    output logic [PAYLOAD_W-1:0] vxm_eastbound_in,
    output logic                 vxm_eastbound_valid_in,

    output logic [PAYLOAD_W-1:0] mem0_westbound_in,
    output logic                 mem0_westbound_valid_in,
    output logic [PAYLOAD_W-1:0] mem0_eastbound_in,
    output logic                 mem0_eastbound_valid_in,

    output logic [PAYLOAD_W-1:0] mem1_westbound_in,
    output logic                 mem1_westbound_valid_in,
    output logic [PAYLOAD_W-1:0] mem1_eastbound_in,
    output logic                 mem1_eastbound_valid_in
);

    logic [PAYLOAD_W-1:0] mxm_payload_e;
    logic                 mxm_valid_e;
    
    logic mxm_west_en;
    logic sxm_west_en; 
    logic mem0_west_en;
    logic vxm_west_en;

    logic sxm_east_en;
    logic mem0_east_en;
    logic vxm_east_en;
    logic mem1_east_en;

    // Per-unit ingress capture registers.
    logic [PAYLOAD_W-1:0] mxm_w_payload_reg;
    logic                 mxm_w_valid_reg;
    logic [PAYLOAD_W-1:0] sxm_w_payload_reg;
    logic                 sxm_w_valid_reg;
    logic [PAYLOAD_W-1:0] mem0_w_payload_reg;
    logic                 mem0_w_valid_reg;
    logic [PAYLOAD_W-1:0] vxm_w_payload_reg;
    logic                 vxm_w_valid_reg;

    logic [PAYLOAD_W-1:0] sxm_e_payload_reg;
    logic                 sxm_e_valid_reg;
    logic [PAYLOAD_W-1:0] mem0_e_payload_reg;
    logic                 mem0_e_valid_reg;
    logic [PAYLOAD_W-1:0] vxm_e_payload_reg;
    logic                 vxm_e_valid_reg;
    logic [PAYLOAD_W-1:0] mem1_e_payload_reg;
    logic                 mem1_e_valid_reg;

    mxm_eastbound_adapter #(
        .MXM_SIZE(MXM_SIZE),
        .PAYLOAD_W(PAYLOAD_W)
    ) u_mxm_to_eastbound (
        .mxm_out(mxm_out),
        .mxm_row_sel(mxm_e_row_sel),
        .mxm_col_sel(mxm_e_col_sel),
        .mxm_valid_in(mxm_e_valid_in),
        .mxm_payload(mxm_payload_e),
        .mxm_valid(mxm_valid_e)
    );

    // Shared westbound bus mux.
    westbound_bus #(
        .PAYLOAD_W(PAYLOAD_W)
    ) u_westbound_bus (
        .producer_sel(westbound_sel),
        .sxm_payload(sxm_payload_w),
        .sxm_valid(sxm_valid_w),
        .mem0_payload(mem0_payload_w),
        .mem0_valid(mem0_valid_w),
        .vxm_payload(vxm_payload_w),
        .vxm_valid(vxm_valid_w),
        .mem1_payload(mem1_payload_w),
        .mem1_valid(mem1_valid_w),
        .westbound_payload(westbound_payload),
        .westbound_valid(westbound_valid)
    );

    //consumer logic for west
    westbound_consumer_decode u_westbound_consumer_decode (
        .consumer_sel (westbound_consumer_sel),
        .westbound_valid (westbound_valid),
        .mxm_west_en(mxm_west_en),
        .sxm_west_en(sxm_west_en),
        .mem0_west_en(mem0_west_en),
        .vxm_west_en(vxm_west_en)
    );

    //consumer logic for east 
    eastbound_consumer_decode u_eastbound_consumer_decode(
        .consumer_sel(eastbound_consumer_sel),
        .eastbound_valid(eastbound_valid),
        .sxm_east_en(sxm_east_en),
        .mem0_east_en(mem0_east_en),
        .vxm_east_en(vxm_east_en),
        .mem1_east_en(mem1_east_en)
    );

    // Capture bus payload into per-unit ingress registers when selected.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxm_w_payload_reg  <= '0;
            mxm_w_valid_reg    <= 1'b0;
            sxm_w_payload_reg  <= '0;
            sxm_w_valid_reg    <= 1'b0;
            mem0_w_payload_reg <= '0;
            mem0_w_valid_reg   <= 1'b0;
            vxm_w_payload_reg  <= '0;
            vxm_w_valid_reg    <= 1'b0;

            sxm_e_payload_reg  <= '0;
            sxm_e_valid_reg    <= 1'b0;
            mem0_e_payload_reg <= '0;
            mem0_e_valid_reg   <= 1'b0;
            vxm_e_payload_reg  <= '0;
            vxm_e_valid_reg    <= 1'b0;
            mem1_e_payload_reg <= '0;
            mem1_e_valid_reg   <= 1'b0;
        end else begin
            // Valid is a one-cycle capture pulse; payload holds last captured word.
            mxm_w_valid_reg  <= mxm_west_en  && westbound_valid;
            sxm_w_valid_reg  <= sxm_west_en  && westbound_valid;
            mem0_w_valid_reg <= mem0_west_en && westbound_valid;
            vxm_w_valid_reg  <= vxm_west_en  && westbound_valid;

            sxm_e_valid_reg  <= sxm_east_en  && eastbound_valid;
            mem0_e_valid_reg <= mem0_east_en && eastbound_valid;
            vxm_e_valid_reg  <= vxm_east_en  && eastbound_valid;
            mem1_e_valid_reg <= mem1_east_en && eastbound_valid;

            if (mxm_west_en && westbound_valid)
                mxm_w_payload_reg <= westbound_payload;
            if (sxm_west_en && westbound_valid)
                sxm_w_payload_reg <= westbound_payload;
            if (mem0_west_en && westbound_valid)
                mem0_w_payload_reg <= westbound_payload;
            if (vxm_west_en && westbound_valid)
                vxm_w_payload_reg <= westbound_payload;

            if (sxm_east_en && eastbound_valid)
                sxm_e_payload_reg <= eastbound_payload;
            if (mem0_east_en && eastbound_valid)
                mem0_e_payload_reg <= eastbound_payload;
            if (vxm_east_en && eastbound_valid)
                vxm_e_payload_reg <= eastbound_payload;
            if (mem1_east_en && eastbound_valid)
                mem1_e_payload_reg <= eastbound_payload;
        end
    end

    // Safety checks: decoder outputs should be one-hot-or-zero.
    always_comb begin
        assert ($onehot0({mxm_west_en, sxm_west_en, mem0_west_en, vxm_west_en}))
            else $error("Invalid westbound consumer decode: more than one enable asserted");
        assert ($onehot0({sxm_east_en, mem0_east_en, vxm_east_en, mem1_east_en}))
            else $error("Invalid eastbound consumer decode: more than one enable asserted");
    end

    // Shared eastbound bus mux.
    eastbound_bus #(
        .PAYLOAD_E(PAYLOAD_W)
    ) u_eastbound_bus (
        .producer_sel(eastbound_sel),
        .mxm_payload_e(mxm_payload_e),
        .mxm_valid_e(mxm_valid_e),
        .vxm_payload_e(vxm_payload_e),
        .vxm_valid_e(vxm_valid_e),
        .sxm_payload_e(sxm_payload_e),
        .sxm_valid_e(sxm_valid_e),
        .mem0_payload_e(mem0_payload_e),
        .mem0_valid_e(mem0_valid_e),
        .eastbound_payload(eastbound_payload),
        .eastbound_valid(eastbound_valid)
    );

    // Present captured ingress words to unit-side tap outputs.
    assign mxm_westbound_in       = mxm_w_payload_reg;
    assign mxm_westbound_valid_in = mxm_w_valid_reg;
    assign mxm_eastbound_in       = eastbound_payload;
    assign mxm_eastbound_valid_in = eastbound_valid;

    assign sxm_westbound_in       = sxm_w_payload_reg;
    assign sxm_westbound_valid_in = sxm_w_valid_reg;
    assign sxm_eastbound_in       = sxm_e_payload_reg;
    assign sxm_eastbound_valid_in = sxm_e_valid_reg;

    assign vxm_westbound_in       = vxm_w_payload_reg;
    assign vxm_westbound_valid_in = vxm_w_valid_reg;
    assign vxm_eastbound_in       = vxm_e_payload_reg;
    assign vxm_eastbound_valid_in = vxm_e_valid_reg;

    assign mem0_westbound_in       = mem0_w_payload_reg;
    assign mem0_westbound_valid_in = mem0_w_valid_reg;
    assign mem0_eastbound_in       = mem0_e_payload_reg;
    assign mem0_eastbound_valid_in = mem0_e_valid_reg;

    assign mem1_westbound_in       = westbound_payload;
    assign mem1_westbound_valid_in = westbound_valid;
    assign mem1_eastbound_in       = mem1_e_payload_reg;
    assign mem1_eastbound_valid_in = mem1_e_valid_reg;

endmodule
