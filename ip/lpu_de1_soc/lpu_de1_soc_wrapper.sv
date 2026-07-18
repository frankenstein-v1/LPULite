`timescale 1ns/1ps

module lpu_de1_soc_wrapper (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] avs_address,
    input  logic        avs_read,
    input  logic        avs_write,
    input  logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata,
    output logic        avs_waitrequest
);
    localparam logic [1:0] EXT_TARGET_MEM0 = 2'd0;
    localparam logic [1:0] EXT_TARGET_MEM1 = 2'd1;
    localparam logic [1:0] EXT_TARGET_IMEM = 2'd2;
    localparam logic [1:0] EXT_TARGET_CTRL = 2'd3;

    logic        run_enable;
    logic        pc_load_en;
    logic [31:0] pc_load_value;
    logic [31:0] cycle_counter;

    logic        ext_en;
    logic        ext_write;
    logic [1:0]  ext_target;
    logic [31:0] ext_addr;
    logic [95:0] ext_wdata;
    logic [95:0] ext_rdata;
    logic [95:0] ext_rdata_latched;

    logic [95:0] imem_assembly;
    logic [71:0] mem0_assembly;
    logic [71:0] mem1_assembly;

    wire [13:0] imem_word = avs_address[13:2];
    wire [13:0] mem0_word = (avs_address - 16'h4000) >> 2;
    wire [13:0] mem1_word = (avs_address - 16'h8000) >> 2;

    integer row_index;
    integer lane_index;

    assign avs_waitrequest = 1'b0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_enable       <= 1'b0;
            pc_load_en       <= 1'b0;
            pc_load_value    <= 32'd0;
            ext_en           <= 1'b0;
            ext_write        <= 1'b0;
            ext_target       <= EXT_TARGET_CTRL;
            ext_addr         <= 32'd0;
            ext_wdata        <= 96'd0;
            ext_rdata_latched <= 96'd0;
            imem_assembly    <= 96'd0;
            mem0_assembly    <= 72'd0;
            mem1_assembly    <= 72'd0;
            avs_readdata     <= 32'd0;
        end else begin
            ext_en <= 1'b0;
            ext_write <= 1'b0;
            pc_load_en <= 1'b0;
            ext_rdata_latched <= ext_rdata;

            if (avs_write) begin
                if (avs_address == 16'hc000) begin
                    run_enable <= avs_writedata[0];
                end else if (avs_address == 16'hc004) begin
                    pc_load_value <= avs_writedata;
                end else if (avs_address == 16'hc008) begin
                    pc_load_en <= avs_writedata[0];
                end else if (avs_address < 16'h4000) begin
                    row_index = imem_word / 3;
                    lane_index = imem_word % 3;
                    unique case (lane_index)
                        0: imem_assembly[31:0] <= avs_writedata;
                        1: imem_assembly[63:32] <= avs_writedata;
                        default: begin
                            ext_en <= 1'b1;
                            ext_write <= 1'b1;
                            ext_target <= EXT_TARGET_IMEM;
                            ext_addr <= row_index[31:0];
                            ext_wdata <= {avs_writedata, imem_assembly[63:0]};
                        end
                    endcase
                end else if (avs_address < 16'h8000) begin
                    row_index = mem0_word / 3;
                    lane_index = mem0_word % 3;
                    unique case (lane_index)
                        0: mem0_assembly[31:0] <= avs_writedata;
                        1: mem0_assembly[63:32] <= avs_writedata;
                        default: begin
                            ext_en <= 1'b1;
                            ext_write <= 1'b1;
                            ext_target <= EXT_TARGET_MEM0;
                            ext_addr <= row_index[31:0];
                            ext_wdata <= {24'd0, avs_writedata[7:0], mem0_assembly[63:0]};
                        end
                    endcase
                end else if (avs_address < 16'hc000) begin
                    row_index = mem1_word / 3;
                    lane_index = mem1_word % 3;
                    unique case (lane_index)
                        0: mem1_assembly[31:0] <= avs_writedata;
                        1: mem1_assembly[63:32] <= avs_writedata;
                        default: begin
                            ext_en <= 1'b1;
                            ext_write <= 1'b1;
                            ext_target <= EXT_TARGET_MEM1;
                            ext_addr <= row_index[31:0];
                            ext_wdata <= {24'd0, avs_writedata[7:0], mem1_assembly[63:0]};
                        end
                    endcase
                end
            end

            if (avs_read) begin
                if (avs_address == 16'hc000) begin
                    avs_readdata <= {31'd0, run_enable};
                end else if (avs_address == 16'hc004) begin
                    avs_readdata <= pc_load_value;
                end else if (avs_address == 16'hc00c) begin
                    avs_readdata <= cycle_counter;
                end else if (avs_address < 16'h4000) begin
                    row_index = imem_word / 3;
                    lane_index = imem_word % 3;
                    ext_en <= 1'b1;
                    ext_write <= 1'b0;
                    ext_target <= EXT_TARGET_IMEM;
                    ext_addr <= row_index[31:0];
                    unique case (lane_index)
                        0: avs_readdata <= ext_rdata_latched[31:0];
                        1: avs_readdata <= ext_rdata_latched[63:32];
                        default: avs_readdata <= ext_rdata_latched[95:64];
                    endcase
                end else if (avs_address < 16'h8000) begin
                    row_index = mem0_word / 3;
                    lane_index = mem0_word % 3;
                    ext_en <= 1'b1;
                    ext_write <= 1'b0;
                    ext_target <= EXT_TARGET_MEM0;
                    ext_addr <= row_index[31:0];
                    unique case (lane_index)
                        0: avs_readdata <= ext_rdata_latched[31:0];
                        1: avs_readdata <= ext_rdata_latched[63:32];
                        default: avs_readdata <= {24'd0, ext_rdata_latched[71:64]};
                    endcase
                end else if (avs_address < 16'hc000) begin
                    row_index = mem1_word / 3;
                    lane_index = mem1_word % 3;
                    ext_en <= 1'b1;
                    ext_write <= 1'b0;
                    ext_target <= EXT_TARGET_MEM1;
                    ext_addr <= row_index[31:0];
                    unique case (lane_index)
                        0: avs_readdata <= ext_rdata_latched[31:0];
                        1: avs_readdata <= ext_rdata_latched[63:32];
                        default: avs_readdata <= {24'd0, ext_rdata_latched[71:64]};
                    endcase
                end else begin
                    avs_readdata <= 32'd0;
                end
            end
        end
    end

    lpu #(
        .RMSNORM_CHUNKS(4),
        .SOFTMAX_CHUNKS(8)
    ) u_lpu (
        .clk(clk),
        .rst_n(rst_n),
        .run_en(run_enable),
        .pc_load_en(pc_load_en),
        .pc_load_value(pc_load_value),
        .ext_en(ext_en),
        .ext_write(ext_write),
        .ext_target(ext_target),
        .ext_addr(ext_addr),
        .ext_wdata(ext_wdata),
        .ext_rdata(ext_rdata),
        .cycle_counter(cycle_counter)
    );
endmodule
