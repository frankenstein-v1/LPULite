module de1_soc_top (
    input  logic        CLOCK_50,  // PIN_AF14 (50MHz clock input)
    input  logic [0:0]  KEY,       // PIN_AJ4 (Active-low pushbutton as Reset)
    output logic [0:0]  LEDR       // PIN_V16 (Diagnostic LED)
);
    logic [31:0] jtag_address, jtag_writedata, jtag_readdata;
    logic [3:0]  jtag_byteenable;
    logic        jtag_read, jtag_write, jtag_waitrequest, jtag_readdatavalid;

    platform_designer_system u_qsys (
        .clk_clk       (CLOCK_50),
        .reset_reset_n (KEY[0]),
        .lpu_avalon_address       (jtag_address),
        .lpu_avalon_read          (jtag_read),
        .lpu_avalon_write         (jtag_write),
        .lpu_avalon_writedata     (jtag_writedata),
        .lpu_avalon_byteenable    (jtag_byteenable),
        .lpu_avalon_readdata      (jtag_readdata),
        .lpu_avalon_waitrequest   (jtag_waitrequest),
        .lpu_avalon_readdatavalid (jtag_readdatavalid)
    );
    lpu_de1_soc_wrapper u_lpu_avalon (
        .clk             (CLOCK_50),
        .rst_n           (KEY[0]),
        .avs_address     (jtag_address[15:0]),
        .avs_read        (jtag_read),
        .avs_write       (jtag_write),
        .avs_writedata   (jtag_writedata),
        .avs_readdata      (jtag_readdata),
        .avs_waitrequest   (jtag_waitrequest),
        .avs_readdatavalid (jtag_readdatavalid)
    );
    assign LEDR[0] = KEY[0];
endmodule
