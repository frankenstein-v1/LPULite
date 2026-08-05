`timescale 1ns/1ps
module lpu_de1_soc_wrapper (
    input logic clk, input logic rst_n,
    input logic [15:0] avs_address, input logic avs_read, input logic avs_write,
    input logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata, output logic avs_waitrequest,
    output logic avs_readdatavalid
);
    localparam logic [1:0] MEM0 = 2'd0, MEM1 = 2'd1, IMEM = 2'd2;
    localparam logic [15:0] CTRL_RUN = 16'hc000;
    localparam logic [15:0] CTRL_PC_LOAD = 16'hc004;
    localparam logic [15:0] CTRL_CYCLES = 16'hc008;
    logic run_enable, pc_load_en, ext_en, ext_write;
    logic [31:0] pc_load_value, cycle_counter;
    logic [1:0] ext_target;
    logic [31:0] ext_addr;
    logic [95:0] ext_wdata, ext_rdata, assembly;
    integer word_index, row_index, lane_index;

    assign avs_waitrequest = 1'b0;
    assign avs_readdatavalid = avs_read;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_enable <= 1'b0; pc_load_en <= 1'b0; pc_load_value <= '0; ext_en <= 1'b0; ext_write <= 1'b0;
            ext_target <= '0; ext_addr <= '0; ext_wdata <= '0; assembly <= '0;
            avs_readdata <= '0;
        end else begin
            ext_en <= 1'b0;
            pc_load_en <= 1'b0;
            if (avs_write) begin
                if (avs_address == CTRL_RUN) run_enable <= avs_writedata[0];
                else if (avs_address == CTRL_PC_LOAD) begin
                    pc_load_value <= avs_writedata;
                    pc_load_en <= 1'b1;
                end
                else begin
                    if (avs_address < 16'h4000) begin word_index = avs_address[13:2]; ext_target <= IMEM; end
                    else if (avs_address < 16'h8000) begin word_index = (avs_address - 16'h4000) >> 2; ext_target <= MEM0; end
                    else begin word_index = (avs_address - 16'h8000) >> 2; ext_target <= MEM1; end
                    row_index = word_index / 3; lane_index = word_index % 3;
                    case (lane_index)
                        0: assembly[31:0] <= avs_writedata;
                        1: assembly[63:32] <= avs_writedata;
                        2: begin
                            ext_en <= 1'b1; ext_write <= 1'b1; ext_addr <= row_index;
                            ext_wdata <= {avs_writedata, assembly[63:0]};
                        end
                    endcase
                end
            end
            if (avs_read) begin
                if (avs_address == CTRL_RUN) avs_readdata <= {31'b0, run_enable};
                else if (avs_address == CTRL_CYCLES) avs_readdata <= cycle_counter;
                else begin
                    if (avs_address < 16'h4000) begin word_index = avs_address[13:2]; ext_target <= IMEM; end
                    else if (avs_address < 16'h8000) begin word_index = (avs_address - 16'h4000) >> 2; ext_target <= MEM0; end
                    else begin word_index = (avs_address - 16'h8000) >> 2; ext_target <= MEM1; end
                    row_index = word_index / 3; lane_index = word_index % 3;
                    ext_en <= 1'b1; ext_write <= 1'b0; ext_addr <= row_index;
                    case (lane_index) 0: avs_readdata <= ext_rdata[31:0];
                      1: avs_readdata <= ext_rdata[63:32];
                      default: avs_readdata <= ext_rdata[95:64]; endcase
                end
            end
        end
    end
    lpu #(
        .RMSNORM_CHUNKS(2),
        .SOFTMAX_CHUNKS(16)
    ) u_lpu (
        .clk(clk), .rst_n(rst_n), .run_en(run_enable),
        .pc_load_en(pc_load_en), .pc_load_value(pc_load_value),
        .ext_en(ext_en), .ext_write(ext_write), .ext_target(ext_target),
        .ext_addr(ext_addr), .ext_wdata(ext_wdata), .ext_rdata(ext_rdata), .cycle_counter(cycle_counter)
    );
endmodule
