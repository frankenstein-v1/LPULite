`timescale 1ns/1ns

`ifndef LPU_PKG_SV
`define LPU_PKG_SV

// 4 lanes of 8-bit data = 32 bits total for the superlane
typedef logic [31:0] superlane_t;
// FP8 lane type (8-bit raw encoding). Use when treating bytes as FP8.
typedef logic [7:0] fp8_t;
localparam int MXM_SIZE = 4;
localparam int MXM_ACC_W = 32;
typedef logic [127:0] mxm_row_t;

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

typedef enum logic [2:0]{
    WC_NONE = 3'd0,
    WC_MXM = 3'd1,
    WC_SXM = 3'd2,
    WC_MEM0 = 3'd3,
    WC_VXM = 3'd4
} westbound_consumer_e;

typedef enum logic [2:0]{
    EC_NONE = 3'd0,
    EC_SXM = 3'd1,
    EC_MEM0 = 3'd2,
    EC_VXM = 3'd3,
    EC_MEM1 = 3'd4
} eastbound_consumer_e;

// 1,280 bytes total per hemisphere.
// 1,280 bytes / 4 bytes per superlane = 320 memory slots
localparam MEM_DEPTH = 320;

`endif
