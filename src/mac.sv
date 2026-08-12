`timescale 1ns/1ps

// One block-scaled fixed-point MAC.
//
// Numeric contract:
//   input_value  = input_i  * 2**input_scale_i
//   weight_value = weight_i * 2**weight_scale_i
//   acc_value    = acc_o    * 2**acc_scale_o
//
// Products can arrive with different scale exponents.  Before accumulation,
// align both the existing accumulator and the new product to the finer (more
// negative) exponent.  Adding the raw integers without this alignment is only
// valid when every product has the same scale, which is not true for block-
// quantized matrix/vector products.
module mac #(
    parameter int INPUT_W   = 8,
    parameter int WEIGHT_W  = 8,
    parameter int PRODUCT_W = INPUT_W + WEIGHT_W,
    parameter int ACC_W     = 32,
    parameter int SCALE_W   = 8,
    parameter bit EXTERNAL_SCALE_CONTROL = 1'b0
) (
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic en,

    input  logic signed [INPUT_W-1:0]  input_i,
    input  logic signed [WEIGHT_W-1:0] weight_i,

    input  logic signed [SCALE_W-1:0] input_scale_i,
    input  logic signed [SCALE_W-1:0] weight_scale_i,
    input  logic                       acc_valid_i,
    input  logic signed [SCALE_W-1:0] aligned_scale_i,
    input  logic        [SCALE_W-1:0] acc_shift_i,
    input  logic        [SCALE_W-1:0] product_shift_i,

    output logic signed [ACC_W-1:0]   acc_o,
    output logic signed [SCALE_W-1:0] acc_scale_o,
    output logic signed [PRODUCT_W-1:0] product_o
);

`ifdef TINYLPU_MXM_MAC_LOGIC_MULT
    (* multstyle = "logic" *) logic signed [PRODUCT_W-1:0] product;
`else
    logic signed [PRODUCT_W-1:0] product;
`endif
    logic signed [ACC_W-1:0]     product_ext;
    logic                        acc_valid;
    logic signed [SCALE_W-1:0]   product_scale;

    localparam int ALIGN_W = ACC_W * 2;
    logic signed [ALIGN_W-1:0] acc_wide;
    logic signed [ALIGN_W-1:0] product_wide;
    logic signed [ALIGN_W-1:0] aligned_acc;
    logic signed [ALIGN_W-1:0] aligned_product;
    logic signed [ALIGN_W-1:0] aligned_sum;
    logic signed [SCALE_W-1:0] aligned_scale;
    integer                     scale_delta;

    localparam logic signed [ALIGN_W-1:0] ACC_MAX_WIDE =
        ({{(ALIGN_W-ACC_W){1'b0}}, 1'b0, {(ACC_W-1){1'b1}}});
    localparam logic signed [ALIGN_W-1:0] ACC_MIN_WIDE =
        ({{(ALIGN_W-ACC_W){1'b1}}, 1'b1, {(ACC_W-1){1'b0}}});

    assign product = $signed(input_i) * $signed(weight_i);
    assign product_ext = {{(ACC_W-PRODUCT_W){product[PRODUCT_W-1]}}, product};
    assign product_scale = $signed(input_scale_i) + $signed(weight_scale_i);

    generate
        if (EXTERNAL_SCALE_CONTROL) begin : g_external_scale_control
            // MXM computes this once because all 64 MAC cells see the same
            // block-scale exponents on a given accumulation cycle.
            always_comb begin
                acc_wide = {{(ALIGN_W-ACC_W){acc_o[ACC_W-1]}}, acc_o};
                product_wide = {{(ALIGN_W-PRODUCT_W){product[PRODUCT_W-1]}}, product};
                aligned_acc = acc_valid_i ? (acc_wide <<< acc_shift_i) : '0;
                aligned_product = product_wide <<< product_shift_i;
                aligned_scale = aligned_scale_i;
                aligned_sum = aligned_acc + aligned_product;
                scale_delta = 0;
            end
        end else begin : g_local_scale_control
            // Standalone/default behavior used by the focused MAC regression.
            always_comb begin
                acc_wide = {{(ALIGN_W-ACC_W){acc_o[ACC_W-1]}}, acc_o};
                product_wide = {{(ALIGN_W-PRODUCT_W){product[PRODUCT_W-1]}}, product};
                aligned_acc = acc_wide;
                aligned_product = product_wide;
                aligned_scale = acc_scale_o;
                scale_delta = 0;

                if (!acc_valid) begin
                    aligned_acc = '0;
                    aligned_scale = product_scale;
                end else if ($signed(product_scale) < $signed(acc_scale_o)) begin
                    scale_delta = $signed(acc_scale_o) - $signed(product_scale);
                    aligned_acc = acc_wide <<< scale_delta;
                    aligned_scale = product_scale;
                end else if ($signed(product_scale) > $signed(acc_scale_o)) begin
                    scale_delta = $signed(product_scale) - $signed(acc_scale_o);
                    aligned_product = product_wide <<< scale_delta;
                end

                aligned_sum = aligned_acc + aligned_product;
            end
        end
    endgenerate

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_o       <= '0;
            acc_scale_o <= '0;
            product_o   <= '0;
            acc_valid   <= 1'b0;
        end else if (clear) begin
            acc_o       <= '0;
            acc_scale_o <= '0;
            product_o   <= '0;
            acc_valid   <= 1'b0;
        end else if (en) begin
            if (aligned_sum > ACC_MAX_WIDE)
                acc_o <= {1'b0, {(ACC_W-1){1'b1}}};
            else if (aligned_sum < ACC_MIN_WIDE)
                acc_o <= {1'b1, {(ACC_W-1){1'b0}}};
            else
                acc_o <= aligned_sum[ACC_W-1:0];
            acc_scale_o <= aligned_scale;
            product_o   <= product;
            acc_valid   <= 1'b1;
        end else begin
            product_o <= '0;
        end
    end

endmodule


// Compatibility wrapper for the current MXM path. New code should instantiate
// mac directly and carry scale metadata explicitly.
module int_mac (
    input  logic              clk,
    input  logic              rst,
    input  logic              en,
    input  logic        [7:0] input_in,
    input  logic              input_is_signed,
    input  logic              weight_load,
    input  logic        [7:0] weight_value,
    input  logic              weight_is_signed,
    output logic signed [19:0] product
);

    logic [7:0]        weight_reg;
    logic              weight_is_signed_reg;
    logic signed [8:0] input_ext;
    logic signed [8:0] weight_ext;
`ifdef TINYLPU_MXM_MAC_LOGIC_MULT
    (* multstyle = "logic" *) logic signed [19:0] product_next;
`else
    logic signed [19:0] product_next;
`endif

    assign input_ext = input_is_signed
        ? $signed({input_in[7], input_in})
        : $signed({1'b0, input_in});

    assign weight_ext = weight_is_signed_reg
        ? $signed({weight_reg[7], weight_reg})
        : $signed({1'b0, weight_reg});

    assign product_next = weight_ext * input_ext;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            weight_reg <= 8'd0;
            weight_is_signed_reg <= 1'b1;
            product <= 20'sd0;
        end else if (weight_load) begin
            weight_reg <= weight_value;
            weight_is_signed_reg <= weight_is_signed;
            product <= 20'sd0;
        end else if (en) begin
            product <= product_next;
        end else begin
            product <= 20'sd0;
        end
    end

endmodule
