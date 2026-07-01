`timescale 1ns/1ns

module lut_layernorm #(
    parameter int LANES      = 4,
    parameter int LANE_W     = 32,
    parameter int LUT_FRAC_W = 16
) (
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*LANE_W-1:0] gamma,
    input  logic [LANES*LANE_W-1:0] beta,
    output logic [LANES*LANE_W-1:0] y_out
);

    // Unpack input lanes as signed values
    logic signed [LANE_W-1:0] x [0:LANES-1];
    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_unpack
            assign x[i] = x_in[i*LANE_W +: LANE_W];
        end
    endgenerate

    // 1. Mean (u) calculation: u = (x0 + x1 + x2 + x3) / 4
    // Using signed addition with extra bits to prevent overflow, then >>> 2
    logic signed [LANE_W+1:0] sum_x;
    assign sum_x = $signed(x[0]) + $signed(x[1]) + $signed(x[2]) + $signed(x[3]);

    logic signed [LANE_W-1:0] u;
    assign u = sum_x >>> 2;

    // 2. Variance (sigma^2) calculation:
    // sigma^2 = ((x0-u)^2 + (x1-u)^2 + (x2-u)^2 + (x3-u)^2) / 4
    logic signed [LANE_W-1:0] diff [0:LANES-1];
    logic signed [2*LANE_W-1:0] sq [0:LANES-1];
    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_diff_sq
            assign diff[i] = x[i] - u;
            assign sq[i]   = diff[i] * diff[i];
        end
    endgenerate

    logic signed [2*LANE_W+1:0] sum_sq;
    assign sum_sq = sq[0] + sq[1] + sq[2] + sq[3];

    logic signed [2*LANE_W-1:0] variance;
    assign variance = sum_sq >>> 2;

    // 3. Lookup Table for Inverse Square Root (1/sqrt(sigma^2))
    // Saturated 16-bit variance index/address
    logic [15:0] sigma2_idx;
    assign sigma2_idx = (variance > 65535) ? 16'd65535 : variance[15:0];

    logic [31:0] lut_rom [0:65535];

    function automatic logic [31:0] calc_inv_sqrt(input int i);
        real val;
        real inv_sqrt_val;
        begin
            if (i == 0) begin
                val = 0.00001; // fallback to prevent division-by-zero
            end else begin
                val = $itor(i);
            end
            inv_sqrt_val = 1.0 / $sqrt(val);
            calc_inv_sqrt = $rtoi(inv_sqrt_val * $itor(1 << LUT_FRAC_W) + 0.5);
        end
    endfunction

    initial begin
        for (int i = 0; i < 65536; i++) begin
            lut_rom[i] = calc_inv_sqrt(i);
        end
    end

    logic [31:0] lut_val;
    assign lut_val = lut_rom[sigma2_idx];

    // 4. Output calculation for each lane: Output = (x_hat * gamma) + beta
    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_output
            logic signed [LANE_W-1:0] gamma_lane;
            logic signed [LANE_W-1:0] beta_lane;
            logic signed [2*LANE_W:0]   x_hat_large;
            logic signed [3*LANE_W-1:0] scaled_large;
            logic signed [3*LANE_W-1:0] scaled_shifted;
            logic signed [LANE_W-1:0]   out_lane;

            assign gamma_lane = gamma[i*LANE_W +: LANE_W];
            assign beta_lane  = beta[i*LANE_W +: LANE_W];

            // Multiply (x - u) by lut_val.
            // Since lut_val is positive, we treat it as positive signed.
            assign x_hat_large = diff[i] * $signed({1'b0, lut_val});

            // Multiply by gamma
            assign scaled_large = x_hat_large * gamma_lane;

            // Shift right by LUT_FRAC_W
            assign scaled_shifted = scaled_large >>> LUT_FRAC_W;

            // Add beta
            assign out_lane = scaled_shifted[LANE_W-1:0] + beta_lane;

            assign y_out[i*LANE_W +: LANE_W] = out_lane;
        end
    endgenerate

endmodule
