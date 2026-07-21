`timescale 1ns/1ps

// Refactor placeholder for the VXM RoPE stage.
//
// The previous implementation used CVFPU FP8 casts and FP32 FMAs. Those units
// have been removed from the active source tree so the fixed/scaled-int RoPE
// implementation can be added without pulling floating-point IP into synthesis.
module vxm_rope #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*8-1:0]      cos_fp8,
    input  logic [LANES*8-1:0]      sin_fp8,
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    output logic                    busy_o
);

    logic unused_rope_tables;

    assign unused_rope_tables = ^{cos_fp8, sin_fp8};
    assign busy_o = start_i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out  <= '0;
            done_o <= 1'b0;
        end else begin
            done_o <= start_i;
            if (start_i)
                y_out <= x_in;
        end
    end

endmodule
