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
    localparam logic [15:0] CTRL_RUN_CYCLES = 16'hc00c;
    localparam logic [15:0] CTRL_SOFT_RESET = 16'hc010;
    logic run_enable, pc_load_en, ext_en, ext_write;
    logic [31:0] pc_load_value, cycle_counter, run_cycles_remaining;
    logic [7:0] soft_reset_count;
    logic lpu_rst_n;
    logic [1:0] ext_target;
    logic [31:0] ext_addr;
    logic [95:0] ext_wdata, ext_rdata, assembly;
    logic [1:0] read_delay;
    logic read_is_mem;
    logic [1:0] read_lane;
    logic [31:0] read_data_pending;
    logic [11:0] decoded_word_index;
    logic [31:0] decoded_row_index;
    logic [1:0] decoded_lane_index;

    function automatic logic [11:0] word_index_from_address(input logic [15:0] address);
        begin
            if (address < 16'h4000) begin
                word_index_from_address = address[13:2];
            end else if (address < 16'h8000) begin
                word_index_from_address = (address - 16'h4000) >> 2;
            end else begin
                word_index_from_address = (address - 16'h8000) >> 2;
            end
        end
    endfunction

    function automatic logic [31:0] row_from_word_index(input logic [11:0] word_index);
        logic [24:0] product;
        begin
            // Exact row decode for packed 3-word rows without inferring a divider.
            product = word_index * 13'd2731;
            row_from_word_index = {20'd0, product[24:13]};
        end
    endfunction

    function automatic logic [1:0] lane_from_word_index(input logic [11:0] word_index);
        logic [31:0] row;
        logic [13:0] row_times_3;
        begin
            row = row_from_word_index(word_index);
            row_times_3 = {2'b0, row[11:0]} + {1'b0, row[11:0], 1'b0};
            lane_from_word_index = word_index - row_times_3[11:0];
        end
    endfunction

    assign avs_waitrequest = (read_delay != 2'd0);
    assign lpu_rst_n = rst_n && (soft_reset_count == 8'd0);

    assign decoded_word_index = word_index_from_address(avs_address);
    assign decoded_row_index = row_from_word_index(decoded_word_index);
    assign decoded_lane_index = lane_from_word_index(decoded_word_index);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_enable <= 1'b0; pc_load_en <= 1'b0; pc_load_value <= '0; ext_en <= 1'b0; ext_write <= 1'b0;
            run_cycles_remaining <= '0;
            soft_reset_count <= '0;
            ext_target <= '0; ext_addr <= '0; ext_wdata <= '0; assembly <= '0;
            avs_readdata <= '0;
            avs_readdatavalid <= 1'b0;
            read_delay <= 2'd0;
            read_is_mem <= 1'b0;
            read_lane <= '0;
            read_data_pending <= '0;
        end else begin
            ext_en <= 1'b0;
            ext_write <= 1'b0;
            pc_load_en <= 1'b0;
            avs_readdatavalid <= 1'b0;

            if (soft_reset_count != 8'd0) begin
                soft_reset_count <= soft_reset_count - 8'd1;
                run_enable <= 1'b0;
                run_cycles_remaining <= 32'd0;
            end

            if (run_cycles_remaining != 32'd0) begin
                if (run_cycles_remaining == 32'd1) begin
                    run_cycles_remaining <= 32'd0;
                    run_enable <= 1'b0;
                end else begin
                    run_cycles_remaining <= run_cycles_remaining - 32'd1;
                end
            end

            if (read_delay != 2'd0) begin
                read_delay <= read_delay - 2'd1;
                if (read_delay == 2'd1) begin
                    avs_readdatavalid <= 1'b1;
                    if (read_is_mem) begin
                        case (read_lane)
                            2'd0: avs_readdata <= ext_rdata[31:0];
                            2'd1: avs_readdata <= ext_rdata[63:32];
                            default: avs_readdata <= ext_rdata[95:64];
                        endcase
                    end else begin
                        avs_readdata <= read_data_pending;
                    end
                end else begin
                    avs_readdatavalid <= 1'b0;
                end
            end else if (avs_write) begin
                if (avs_address == CTRL_RUN) begin
                    run_enable <= avs_writedata[0];
                    run_cycles_remaining <= 32'd0;
                end
                else if (avs_address == CTRL_PC_LOAD) begin
                    pc_load_value <= avs_writedata;
                    pc_load_en <= 1'b1;
                end
                else if (avs_address == CTRL_RUN_CYCLES) begin
                    run_cycles_remaining <= avs_writedata;
                    run_enable <= (avs_writedata != 32'd0);
                end
                else if (avs_address == CTRL_SOFT_RESET) begin
                    run_enable <= 1'b0;
                    run_cycles_remaining <= 32'd0;
                    pc_load_value <= '0;
                    pc_load_en <= 1'b1;
                    soft_reset_count <= (avs_writedata[7:0] == 8'd0) ? 8'd16 : avs_writedata[7:0];
                end
                else begin
                    if (avs_address < 16'h4000) begin ext_target <= IMEM; end
                    else if (avs_address < 16'h8000) begin ext_target <= MEM0; end
                    else begin ext_target <= MEM1; end
                    case (decoded_lane_index)
                        2'd0: assembly[31:0] <= avs_writedata;
                        2'd1: assembly[63:32] <= avs_writedata;
                        2'd2: begin
                            ext_en <= 1'b1; ext_write <= 1'b1; ext_addr <= decoded_row_index;
                            ext_wdata <= {avs_writedata, assembly[63:0]};
                        end
                    endcase
                end
            end else if (avs_read) begin
                if (avs_address == CTRL_RUN) begin
                    read_delay <= 2'd1;
                    read_is_mem <= 1'b0;
                    read_data_pending <= {31'b0, run_enable};
                end else if (avs_address == CTRL_CYCLES) begin
                    read_delay <= 2'd1;
                    read_is_mem <= 1'b0;
                    read_data_pending <= cycle_counter;
                end else if (avs_address == CTRL_RUN_CYCLES) begin
                    read_delay <= 2'd1;
                    read_is_mem <= 1'b0;
                    read_data_pending <= run_cycles_remaining;
                end else if (avs_address == CTRL_SOFT_RESET) begin
                    read_delay <= 2'd1;
                    read_is_mem <= 1'b0;
                    read_data_pending <= {24'b0, soft_reset_count};
                end
                else begin
                    if (avs_address < 16'h4000) begin ext_target <= IMEM; end
                    else if (avs_address < 16'h8000) begin ext_target <= MEM0; end
                    else begin ext_target <= MEM1; end
                    ext_en <= 1'b1; ext_write <= 1'b0; ext_addr <= decoded_row_index;
                    // MEM/IMEM are FPGA block RAMs with registered output.
                    // ext_en/addr are registered by this wrapper, then the
                    // memory captures the read on the following clock, and
                    // its q output is visible after another registered clock.
                    read_delay <= 2'd3;
                    read_is_mem <= 1'b1;
                    read_lane <= decoded_lane_index;
                end
            end
        end
    end
    lpu #(
        .RMSNORM_CHUNKS(2),
        .SOFTMAX_CHUNKS(16)
    ) u_lpu (
        .clk(clk), .rst_n(lpu_rst_n), .run_en(run_enable),
        .pc_load_en(pc_load_en), .pc_load_value(pc_load_value),
        .ext_en(ext_en), .ext_write(ext_write), .ext_target(ext_target),
        .ext_addr(ext_addr), .ext_wdata(ext_wdata), .ext_rdata(ext_rdata), .cycle_counter(cycle_counter)
    );
endmodule
