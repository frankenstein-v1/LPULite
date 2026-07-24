`timescale 1ns/1ps

// Synthesizable pipelined fixed-point RMSNorm unit.
// Accepts 8 lanes x 32-bit signed fixed-point vector and per-lane gamma scaling parameters.
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

    logic [LANES*LANE_W-1:0] lut_y_out;
    logic                    output_valid_q;

    // Instantiate synthesizable combinatorial fixed-point RMSNorm block
    lut_rmsnorm #(
        .LANES(LANES),
        .LANE_W(LANE_W)
    ) lut_inst (
        .x_in(x_in),
        .gamma(gamma),
        .beta(beta),
        .y_out(lut_y_out)
    );

    assign in_ready = !output_valid_q;
    assign done_o   = output_valid_q;
    assign busy_o   = output_valid_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valid_q <= 1'b0;
            y_out          <= '0;
        end else begin
            if (output_valid_q && out_ready) begin
                output_valid_q <= 1'b0;
            end

            if (start_i && in_ready) begin
                y_out          <= lut_y_out;
                output_valid_q <= 1'b1;
            end
        end
    end

endmodule
