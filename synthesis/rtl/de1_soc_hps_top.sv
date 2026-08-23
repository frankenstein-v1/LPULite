module de1_soc_hps_top (
    input  logic        CLOCK_50,
    input  logic [0:0]  KEY,
    output logic [0:0]  LEDR
);
    // LPULite currently does not close timing at the board 50 MHz clock.
    // TimeQuest reports an LPU fabric Fmax of ~11 MHz, so run the entire
    // lightweight-HPS/LPU Platform Designer fabric at 50/8 = 6.25 MHz.
    logic [2:0] qsys_clk_div;
    logic       qsys_clk;

    always_ff @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0]) begin
            qsys_clk_div <= 3'd0;
        end else begin
            qsys_clk_div <= qsys_clk_div + 3'd1;
        end
    end

    assign qsys_clk = qsys_clk_div[2];

    platform_designer_hps_system u_qsys (
        .clk_clk       (qsys_clk),
        .reset_reset_n (KEY[0])
    );

    assign LEDR[0] = KEY[0];
endmodule
