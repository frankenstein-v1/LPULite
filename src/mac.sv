`timescale 1ns/1ps

// One block-scaled fixed-point MAC.
//
// Numeric contract:
//   input_value  = input_i  * 2**input_scale_i
//   weight_value = weight_i * 2**weight_scale_i
//   acc_value    = acc_o    * 2**acc_scale_o
//
// The scale is carried as metadata. This block does not apply the scale to the
// multiply; it accumulates raw signed products and reports the combined scale.
module mac #(
    parameter int INPUT_W   = 8,
    parameter int WEIGHT_W  = 8,
    parameter int PRODUCT_W = INPUT_W + WEIGHT_W,
    parameter int ACC_W     = 32,
    parameter int SCALE_W   = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic en,

    input  logic signed [INPUT_W-1:0]  input_i,
    input  logic signed [WEIGHT_W-1:0] weight_i,

    input  logic signed [SCALE_W-1:0] input_scale_i,
    input  logic signed [SCALE_W-1:0] weight_scale_i,

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

    assign product = $signed(input_i) * $signed(weight_i);
    assign product_ext = {{(ACC_W-PRODUCT_W){product[PRODUCT_W-1]}}, product};

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_o       <= '0;
            acc_scale_o <= '0;
            product_o   <= '0;
        end else if (clear) begin
            acc_o       <= '0;
            acc_scale_o <= '0;
            product_o   <= '0;
        end else if (en) begin
            acc_o       <= acc_o + product_ext;
            acc_scale_o <= input_scale_i + weight_scale_i;
            product_o   <= product;
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
