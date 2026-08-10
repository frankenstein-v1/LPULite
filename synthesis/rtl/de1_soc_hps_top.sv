module de1_soc_hps_top (
    input  logic        CLOCK_50,
    input  logic [0:0]  KEY,
    output logic [0:0]  LEDR
);
    platform_designer_hps_system u_qsys (
        .clk_clk       (CLOCK_50),
        .reset_reset_n (KEY[0])
    );

    assign LEDR[0] = KEY[0];
endmodule
