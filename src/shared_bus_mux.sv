`timescale 1ns/1ps

module shared_bus_mux #(
    parameter int PAYLOAD_W = 32,
    parameter int N_SOURCES = 4,
    parameter int SEL_W = 2
) (
    input  logic [SEL_W-1:0]                  select_i,
    input  logic [N_SOURCES-1:0][PAYLOAD_W-1:0] payload_i,
    input  logic [N_SOURCES-1:0]              valid_i,
    output logic [PAYLOAD_W-1:0]              payload_o,
    output logic                              valid_o
);

    always_comb begin
        payload_o = '0;
        valid_o   = 1'b0;

        if (select_i < N_SOURCES) begin
            payload_o = payload_i[select_i];
            valid_o   = valid_i[select_i];
        end
    end

endmodule
