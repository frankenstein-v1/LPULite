`timescale 1ns/1ps

module mxm_2x2_tb;
  localparam int N = 2;

  logic clk;
  logic rst;
  logic mxm_clear;
  logic mxm_start;

  logic signed [7:0] mxm_act_in [N-1:0];
  logic              wght_load  [N-1:0];
  logic signed [7:0] wght_val   [N-1:0];
  logic signed [31:0] mxm_out   [N-1:0][N-1:0];

  // Flattened probes for easier waveform viewing.
  wire signed [31:0] c00;
  wire signed [31:0] c01;
  wire signed [31:0] c10;
  wire signed [31:0] c11;

  logic signed [7:0] a_mat [N-1:0][N-1:0];
  logic signed [7:0] b_mat [N-1:0][N-1:0];
  int signed expected [N-1:0][N-1:0];

  int r;
  int c;
  int k;
  int errors;

  mxm #(
    .mxm_size(N)
  ) dut (
    .clk(clk),
    .rst(rst),
    .mxm_clear(mxm_clear),
    .mxm_start(mxm_start),
    .mxm_act_in(mxm_act_in),
    .wght_load(wght_load),
    .wght_val(wght_val),
    .mxm_out(mxm_out)
  );

  assign c00 = mxm_out[0][0];
  assign c01 = mxm_out[0][1];
  assign c10 = mxm_out[1][0];
  assign c11 = mxm_out[1][1];

  always #5 clk = ~clk;

  task automatic tick;
    @(posedge clk);
    #1;
  endtask

`ifdef WAVEFORM
  initial begin
    $dumpfile("build/mxm_2x2_tb.vcd");
    $dumpvars(0, mxm_2x2_tb);
  end
`endif

  initial begin
    clk = 1'b0;
    rst = 1'b1;
    mxm_clear = 1'b0;
    mxm_start = 1'b0;

    for (r = 0; r < N; r++) begin
      mxm_act_in[r] = '0;
      wght_load[r] = 1'b0;
      wght_val[r] = '0;
      for (c = 0; c < N; c++) begin
        expected[r][c] = 0;
      end
    end

    // A = [ [1, 2],
    //       [3, 4] ]
    a_mat[0][0] = 8'sd1; a_mat[0][1] = 8'sd2;
    a_mat[1][0] = 8'sd3; a_mat[1][1] = 8'sd4;

    // B = [ [5, 6],
    //       [7, 8] ]
    b_mat[0][0] = 8'sd5; b_mat[0][1] = 8'sd6;
    b_mat[1][0] = 8'sd7; b_mat[1][1] = 8'sd8;

    // Reference C = A x B.
    for (r = 0; r < N; r++) begin
      for (c = 0; c < N; c++) begin
        expected[r][c] = 0;
        for (k = 0; k < N; k++) begin
          expected[r][c] = expected[r][c] + (a_mat[r][k] * b_mat[k][c]);
        end
      end
    end

    repeat (2) tick();
    rst = 1'b0;

    // Clear accumulators once after reset.
    mxm_clear = 1'b1;
    tick();
    mxm_clear = 1'b0;

    // Stream K dimension.
    // Cycle A: load weights for k and provide activations.
    // Cycle B: compute with loaded weights.
    mxm_start = 1'b1;
    for (k = 0; k < N; k++) begin
      for (r = 0; r < N; r++) begin
        mxm_act_in[r] = a_mat[r][k];
      end
      for (c = 0; c < N; c++) begin
        wght_val[c] = b_mat[k][c];
        wght_load[c] = 1'b1;
      end
      tick();

      for (c = 0; c < N; c++) begin
        wght_load[c] = 1'b0;
      end
      tick();
    end

    // Flush final product into accumulator.
    for (r = 0; r < N; r++) begin
      mxm_act_in[r] = '0;
    end
    tick();
    mxm_start = 1'b0;

    errors = 0;
    for (r = 0; r < N; r++) begin
      for (c = 0; c < N; c++) begin
        if ($signed(mxm_out[r][c]) !== expected[r][c]) begin
          $display("ERROR C[%0d][%0d]: got=%0d expected=%0d", r, c, $signed(mxm_out[r][c]), expected[r][c]);
          errors = errors + 1;
        end
      end
    end

    if (errors == 0) begin
      $display("PASS: 2x2 matrix multiply matches expected result.");
    end else begin
      $fatal(1, "FAIL: %0d mismatches in matrix result.", errors);
    end

    #10;
    $finish;
  end
endmodule
