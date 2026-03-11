package lpu_pkg;
    // 4 lanes of 8-bit data = 32 bits total for the superlane
    typedef logic [31:0] superlane_t;
    
    // 1,280 bytes total per hemisphere. 
    // 1,280 bytes / 4 bytes per superlane = 320 memory slots
    localparam MEM_DEPTH = 320;
endpackage