`timescale 1ns/1ps

module de1_lpu_top #(
    parameter int RMSNORM_CHUNKS = 4,
    parameter int SOFTMAX_CHUNKS = 8
) (
    input  logic        CLOCK_50,
    input  logic [3:0]  KEY,
    input  logic [9:0]  SW,
    output logic [9:0]  LEDR,
    output logic [6:0]  HEX0,
    output logic [6:0]  HEX1,
    output logic [6:0]  HEX2,
    output logic [6:0]  HEX3,
    output logic [6:0]  HEX4,
    output logic [6:0]  HEX5
);

    logic [2:0] reset_sync;
    logic       board_rst_n;
    logic       lpu_rst_n;
    logic       lpu_run_en;
    logic [31:0] cycle_counter;

    always_ff @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0]) begin
            reset_sync <= 3'b000;
        end else begin
            reset_sync <= {reset_sync[1:0], 1'b1};
        end
    end

    assign board_rst_n = reset_sync[2];
    assign lpu_rst_n   = board_rst_n && SW[0];
    assign lpu_run_en  = SW[1];

    always_ff @(posedge CLOCK_50 or negedge board_rst_n) begin
        if (!board_rst_n) begin
            cycle_counter <= 32'd0;
        end else if (lpu_run_en) begin
            cycle_counter <= cycle_counter + 32'd1;
        end
    end

    lpu #(
        .RMSNORM_CHUNKS(RMSNORM_CHUNKS),
        .SOFTMAX_CHUNKS(SOFTMAX_CHUNKS)
    ) u_lpu (
        .clk          (CLOCK_50),
        .rst_n        (lpu_rst_n),
        .run_en       (lpu_run_en),
        .pc_load_en   (1'b0),
        .pc_load_value(32'd0),
        .ext_en       (1'b0),
        .ext_write    (1'b0),
        .ext_target   (2'd0),
        .ext_addr     (32'd0),
        .ext_wdata    (96'd0),
        .ext_rdata    (),
        .cycle_counter()
    );

    assign LEDR[0] = SW[0];
    assign LEDR[1] = SW[1];
    assign LEDR[2] = lpu_rst_n;
    assign LEDR[3] = cycle_counter[23];
    assign LEDR[4] = cycle_counter[24];
    assign LEDR[5] = cycle_counter[25];
    assign LEDR[6] = cycle_counter[26];
    assign LEDR[7] = cycle_counter[27];
    assign LEDR[8] = cycle_counter[28];
    assign LEDR[9] = cycle_counter[29];

    sevenseg_hex u_hex0 (.value(cycle_counter[3:0]),   .segments_n(HEX0));
    sevenseg_hex u_hex1 (.value(cycle_counter[7:4]),   .segments_n(HEX1));
    sevenseg_hex u_hex2 (.value(cycle_counter[11:8]),  .segments_n(HEX2));
    sevenseg_hex u_hex3 (.value(cycle_counter[15:12]), .segments_n(HEX3));
    sevenseg_hex u_hex4 (.value(4'h1),                 .segments_n(HEX4));
    sevenseg_hex u_hex5 (.value(4'h0),                 .segments_n(HEX5));

endmodule


module sevenseg_hex (
    input  logic [3:0] value,
    output logic [6:0] segments_n
);
    always_comb begin
        unique case (value)
            4'h0: segments_n = 7'b1000000;
            4'h1: segments_n = 7'b1111001;
            4'h2: segments_n = 7'b0100100;
            4'h3: segments_n = 7'b0110000;
            4'h4: segments_n = 7'b0011001;
            4'h5: segments_n = 7'b0010010;
            4'h6: segments_n = 7'b0000010;
            4'h7: segments_n = 7'b1111000;
            4'h8: segments_n = 7'b0000000;
            4'h9: segments_n = 7'b0010000;
            4'ha: segments_n = 7'b0001000;
            4'hb: segments_n = 7'b0000011;
            4'hc: segments_n = 7'b1000110;
            4'hd: segments_n = 7'b0100001;
            4'he: segments_n = 7'b0000110;
            4'hf: segments_n = 7'b0001110;
            default: segments_n = 7'b1111111;
        endcase
    end
endmodule
