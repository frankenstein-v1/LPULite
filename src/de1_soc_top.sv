module de1_soc_top (
    input  logic        CLOCK_50,  // PIN_AF14 (50MHz clock input)
    input  logic [0:0]  KEY,       // PIN_AJ4 (Active-low pushbutton as Reset)
    output logic [0:0]  LEDR       // PIN_V16 (Diagnostic LED)
);
    platform_designer_system u_qsys (
        .clk_clk       (CLOCK_50),
        .reset_reset_n (KEY[0])
    );
    assign LEDR[0] = KEY[0];
endmodule
