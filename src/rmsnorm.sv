`timescale 1ns/1ps

// Refactor placeholder for chunked RMSNorm.
//
// The previous implementation depended on CVFPU FP32 add/div/sqrt/FMA units.
// This shell keeps the VXM interface intact while the fixed/scaled-int RMSNorm
// datapath is rebuilt.
module rmsnorm #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32,
    parameter int CHUNKS = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta,
    output logic                    in_ready,
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    input  logic                    out_ready,
    output logic                    busy_o
);

    localparam logic [31:0] CHUNKS_WORD = CHUNKS;

    logic output_valid_q;
    logic unused_inputs;

    assign unused_inputs = ^{gamma, beta, CHUNKS_WORD};
    assign in_ready = !output_valid_q;
    assign done_o = output_valid_q;
    assign busy_o = output_valid_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valid_q <= 1'b0;
            y_out          <= '0;
        end else begin
            if (output_valid_q && out_ready)
                output_valid_q <= 1'b0;

            if (start_i && in_ready) begin
                y_out          <= x_in;
                output_valid_q <= 1'b1;
            end
        end
    end

endmodule
