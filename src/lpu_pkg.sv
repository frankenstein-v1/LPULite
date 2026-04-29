package lpu_pkg;
    // 4 lanes of 8-bit data = 32 bits total for the superlane
    typedef logic [31:0] superlane_t;

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
endpackage
