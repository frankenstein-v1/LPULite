`timescale 1ns/1ns

`ifndef LPU_PKG_SV
`define LPU_PKG_SV

// 4 lanes of 8-bit data = 32 bits total for the superlane
typedef logic [31:0] superlane_t;
// FP8 lane type (8-bit raw encoding). Use when treating bytes as FP8.
typedef logic [7:0] fp8_t;
typedef logic [7:0] fp8_lane_t;
typedef logic [31:0] packed_fp8_row_t;
typedef logic [7:0] fp8_row_scale_t;
localparam int MXM_SIZE = 4;
localparam int MXM_ACC_W = 32;
typedef logic [127:0] mxm_row_t;
typedef logic [63:0] mem_row_t;

// Memory format for one quantized FP8 row:
//   [31:0]   packed 4-lane FP8 row
//   [39:32]  shared row-scale metadata
//   [63:40]  reserved for future metadata/expansion
typedef struct packed {
    logic [23:0]     reserved;
    fp8_row_scale_t  row_scale;
    packed_fp8_row_t packed_row;
} fp8_row_mem_t;

function automatic fp8_row_mem_t make_fp8_row_mem(
    input packed_fp8_row_t packed_row,
    input logic [31:0]     row_scale_word
);
    fp8_row_mem_t encoded_row;
    begin
        encoded_row = '0;
        encoded_row.packed_row = packed_row;
        encoded_row.row_scale = row_scale_word[7:0];
        make_fp8_row_mem = encoded_row;
    end
endfunction

function automatic packed_fp8_row_t fp8_row_mem_packed(
    input mem_row_t raw_row
);
    fp8_row_mem_t decoded_row;
    begin
        decoded_row = fp8_row_mem_t'(raw_row);
        fp8_row_mem_packed = decoded_row.packed_row;
    end
endfunction

function automatic fp8_row_scale_t fp8_row_mem_scale(
    input mem_row_t raw_row
);
    fp8_row_mem_t decoded_row;
    begin
        decoded_row = fp8_row_mem_t'(raw_row);
        fp8_row_mem_scale = decoded_row.row_scale;
    end
endfunction

function automatic mem_row_t eastbound_to_mem_row(
    input mxm_row_t eastbound_row
);
    begin
        eastbound_to_mem_row = eastbound_row[$bits(mem_row_t)-1:0];
    end
endfunction

function automatic mxm_row_t mem_row_to_eastbound(
    input mem_row_t raw_row
);
    mxm_row_t widened_row;
    begin
        widened_row = '0;
        widened_row[$bits(mem_row_t)-1:0] = raw_row;
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

// 1,280 bytes total per hemisphere.
// 1,280 bytes / 4 bytes per superlane = 320 memory slots
localparam MEM_DEPTH = 320;

`endif
