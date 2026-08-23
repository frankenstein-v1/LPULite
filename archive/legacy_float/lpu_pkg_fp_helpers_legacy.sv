`timescale 1ns/1ns

`ifndef LPU_PKG_SV
`define LPU_PKG_SV

// 8 lanes of 8-bit data = 64 bits total for the superlane.
typedef logic [63:0] superlane_t;
// FP8 lane type (8-bit raw encoding). Use when treating bytes as FP8.
typedef logic [7:0] fp8_t;
typedef logic [7:0] fp8_lane_t;
typedef logic [63:0] packed_fp8_row_t;
typedef logic [7:0] fp8_row_scale_t;
localparam int MXM_SIZE = 8;
localparam int MXM_ACC_W = 32;
typedef logic [255:0] mxm_row_t;

// Fixed-point bus row formats:
//   westbound: 8 x int8 lanes plus one shared row scale
//   eastbound: 8 x int32 lanes plus one shared row scale
typedef logic signed [7:0]  fixed8_lane_t;
typedef logic signed [31:0] fixed32_lane_t;
typedef logic signed [7:0]  fixed_row_scale_t;
typedef logic [63:0]        fixed8_row_data_t;
typedef logic [255:0]       fixed32_row_data_t;
typedef logic [71:0]        westbound_row_t;
typedef logic [263:0]       eastbound_row_t;
typedef westbound_row_t     mem_row_t;

typedef struct packed {
    fixed_row_scale_t row_scale;
    fixed8_row_data_t packed_row;
} westbound_fixed_row_t;

typedef struct packed {
    fixed_row_scale_t  row_scale;
    fixed32_row_data_t packed_row;
} eastbound_fixed_row_t;

// Memory format for one quantized FP8 row:
//   [63:0]   packed 8-lane FP8 row
//   [71:64]  shared row-scale metadata
typedef struct packed {
    fp8_row_scale_t  row_scale;
    packed_fp8_row_t packed_row;
} fp8_row_mem_t;

function automatic mem_row_t make_fp8_row_mem(
    input packed_fp8_row_t packed_row,
    input logic [31:0]     row_scale_word
);
    mem_row_t encoded_row;
    begin
        encoded_row = '0;
        encoded_row[63:0] = packed_row;
        encoded_row[71:64] = row_scale_word[7:0];
        make_fp8_row_mem = encoded_row;
    end
endfunction

function automatic westbound_row_t make_westbound_row(
    input fixed8_row_data_t packed_row,
    input logic [31:0]      row_scale_word
);
    begin
        make_westbound_row = '0;
        make_westbound_row[63:0] = packed_row;
        make_westbound_row[71:64] = row_scale_word[7:0];
    end
endfunction

function automatic fixed8_row_data_t westbound_row_data(
    input westbound_row_t raw_row
);
    begin
        westbound_row_data = raw_row[63:0];
    end
endfunction

function automatic fixed_row_scale_t westbound_row_scale(
    input westbound_row_t raw_row
);
    begin
        westbound_row_scale = raw_row[71:64];
    end
endfunction

function automatic eastbound_row_t make_eastbound_row(
    input fixed32_row_data_t packed_row,
    input logic [31:0]       row_scale_word
);
    begin
        make_eastbound_row = '0;
        make_eastbound_row[255:0] = packed_row;
        make_eastbound_row[263:256] = row_scale_word[7:0];
    end
endfunction

function automatic fixed32_row_data_t eastbound_row_data(
    input eastbound_row_t raw_row
);
    begin
        eastbound_row_data = raw_row[255:0];
    end
endfunction

function automatic fixed_row_scale_t eastbound_row_scale(
    input eastbound_row_t raw_row
);
    begin
        eastbound_row_scale = raw_row[263:256];
    end
endfunction

function automatic packed_fp8_row_t fp8_row_mem_packed8(
    input mem_row_t raw_row
);
    begin
        fp8_row_mem_packed8 = raw_row[63:0];
    end
endfunction

function automatic superlane_t fp8_row_mem_packed(
    input mem_row_t raw_row
);
    begin
        fp8_row_mem_packed = fp8_row_mem_packed8(raw_row);
    end
endfunction

function automatic fp8_row_scale_t fp8_row_mem_scale(
    input mem_row_t raw_row
);
    begin
        fp8_row_mem_scale = raw_row[71:64];
    end
endfunction

function automatic mem_row_t truncate_eastbound_to_mem_row(
    input eastbound_row_t eastbound_row
);
    begin
        // Temporary bridge only: real MXM result stores should go through an
        // int32-to-int8 requantizer before reaching memory.
        truncate_eastbound_to_mem_row = make_fp8_row_mem(eastbound_row[63:0], eastbound_row[263:256]);
    end
endfunction

function automatic eastbound_row_t mem_row_to_eastbound(
    input mem_row_t raw_row
);
    eastbound_row_t widened_row;
    begin
        widened_row = '0;
        widened_row[63:0] = fp8_row_mem_packed8(raw_row);
        widened_row[263:256] = fp8_row_mem_scale(raw_row);
        mem_row_to_eastbound = widened_row;
    end
endfunction

// Producer IDs for the shared westbound bus select signal.
typedef enum logic [2:0] {
    WB_NONE = 3'd0,
    WB_SXM  = 3'd1,
    WB_MEM0 = 3'd2,
    WB_VXM  = 3'd3,
    WB_MEM1 = 3'd4
} westbound_producer_e;

typedef enum logic [2:0] {
    EB_NONE = 3'd0,
    EB_MXM  = 3'd1,
    EB_SXM  = 3'd2,
    EB_MEM0 = 3'd3,
    EB_VXM  = 3'd4
} eastbound_producer_e;

typedef enum logic [2:0] {
    WC_NONE = 3'd0,
    WC_MXM  = 3'd1,
    WC_SXM  = 3'd2,
    WC_MEM0 = 3'd3,
    WC_VXM  = 3'd4
} westbound_consumer_e;

typedef enum logic [2:0] {
    EC_NONE = 3'd0,
    EC_SXM  = 3'd1,
    EC_MEM0 = 3'd2,
    EC_VXM  = 3'd3,
    EC_MEM1 = 3'd4
} eastbound_consumer_e;

typedef enum logic [1:0] {
    QUANT_SIGNED_INT8 = 2'd0,
    QUANT_SOFTMAX_U8  = 2'd1,
    QUANT_FP8_E5M2    = 2'd2
} quant_mode_e;

// Expanded scratch/KV-cache storage for model work.
// Each memory has 32768 rows. At the current 72-bit mem_row_t width,
// that is 288 KiB per memory, or 576 KiB across mem0 + mem1.
localparam int MEM_DEPTH  = 32768;
localparam int MEM_ADDR_W = $clog2(MEM_DEPTH);

// Current LPULite GQA decode cache map.
// K cache lives in MEM0 and V cache lives in MEM1 at the same per-layer offset:
//   addr = layer * KV_CACHE_LAYER_ROWS + token_pos * KV_CACHE_ROWS_PER_TOKEN + kv_head
// With dim=64, heads=8, kv_heads=4, head_dim=8, each token stores four 8-lane rows.
localparam int MODEL_MAX_SEQ_LEN        = 512;
localparam int MODEL_LAYERS             = 5;
localparam int MODEL_KV_HEADS           = 4;
localparam int MODEL_HEAD_DIM           = 8;
localparam int KV_CACHE_ROWS_PER_TOKEN  = MODEL_KV_HEADS;
localparam int KV_CACHE_LAYER_ROWS      = MODEL_MAX_SEQ_LEN * KV_CACHE_ROWS_PER_TOKEN;
localparam int KV_CACHE_TOTAL_ROWS      = MODEL_LAYERS * KV_CACHE_LAYER_ROWS;

`ifndef SYNTHESIS
function automatic real fp32_to_real(input logic [31:0] fp32);
    logic [63:0] r_bits;
    logic        sign;
    logic [7:0]  exp;
    logic [22:0] frac;
    logic [10:0] exp_r;
    logic [51:0] frac_r;
    begin
        sign = fp32[31];
        exp  = fp32[30:23];
        frac = fp32[22:0];

        if (exp == 8'd0) begin
            if (frac == 23'd0) begin
                exp_r  = 11'd0;
                frac_r = 52'd0;
            end else begin
                exp_r  = 11'd1023 - 11'd126;
                frac_r = {frac, 29'd0};
            end
        end else if (exp == 8'hFF) begin
            exp_r  = 11'h7FF;
            frac_r = {frac, 29'd0};
        end else begin
            exp_r  = exp - 8'd127 + 11'd1023;
            frac_r = {frac, 29'd0};
        end

        r_bits = {sign, exp_r, frac_r};
        fp32_to_real = $bitstoreal(r_bits);
    end
endfunction

function automatic logic [31:0] real_to_fp32(input real r);
    logic [63:0] r_bits;
    logic        sign;
    logic [10:0] exp_r;
    logic [51:0] frac_r;
    logic [7:0]  exp;
    logic [22:0] frac;
    begin
        r_bits = $realtobits(r);
        sign   = r_bits[63];
        exp_r  = r_bits[62:52];
        frac_r = r_bits[51:0];

        if (exp_r == 11'd0) begin
            exp  = 8'd0;
            frac = 23'd0;
        end else if (exp_r == 11'h7FF) begin
            exp  = 8'hFF;
            frac = frac_r[51:29];
        end else begin
            integer exp_unbiased;
            exp_unbiased = exp_r - 11'd1023;
            if (exp_unbiased > 127) begin
                exp  = 8'hFF;
                frac = 23'd0;
            end else if (exp_unbiased < -126) begin
                exp  = 8'd0;
                frac = 23'd0;
            end else begin
                exp  = exp_unbiased + 127;
                frac = frac_r[51:29];
            end
        end

        real_to_fp32 = {sign, exp, frac};
    end
endfunction
`endif

`endif
