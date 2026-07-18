`timescale 1ns/1ps
`include "lpu_pkg.sv"

module lpu #(
    parameter int RMSNORM_CHUNKS = 8,
    parameter int SOFTMAX_CHUNKS = 64
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        run_en,
    input  logic        pc_load_en,
    input  logic [31:0] pc_load_value,

    input  logic        ext_en,
    input  logic        ext_write,
    input  logic [1:0]  ext_target,
    input  logic [31:0] ext_addr,
    input  logic [95:0] ext_wdata,
    output logic [95:0] ext_rdata,
    output logic [31:0] cycle_counter
);

//icu mem outputs
logic mem0_read_en;
logic mem0_write_en;
logic [MEM_ADDR_W-1:0] mem0_addr;
logic mem0_read_en_eff;
logic [MEM_ADDR_W-1:0] mem0_addr_eff;
logic mem1_read_en;
logic mem1_write_en;
logic [MEM_ADDR_W-1:0] mem1_addr;
logic mem1_read_en_eff;
logic [MEM_ADDR_W-1:0] mem1_addr_eff;

//icu sxm outputs
logic [11:0] sxm_opcode_input;
logic [11:0] sxm_opcode_weight;
logic sxm_load_from_west_ctrl;

//icxu vxm outs
logic [3:0] vxm_ctrl;
logic vxm_data_sel;
logic [2:0] vxm_operand_sel;

//icu busses
logic [2:0] westbound_sel;
logic [2:0] eastbound_sel;
logic [2:0] westbound_consumer_sel;
logic  [2:0] eastbound_consumer_sel;

//icu mxm outs
logic [1:0] mxm_ingress_mode;
logic mxm_start;
logic mxm_clear;
logic [2:0] mxm_e_row_sel;
logic [2:0] mxm_e_col_sel;
logic mxm_e_valid_in;
logic mxm_input_is_signed;
logic mxm_wght_is_signed;
logic mxm_use_fp;
logic fp_quant_mode;
logic [1:0] mem_store_fmt;
logic vxm_rmsnorm_en;
logic vxm_rope_en;
logic [2:0] vxm_residual_op;

localparam logic [2:0] VXM_OPERAND_DATA     = 3'd0;
localparam logic [2:0] VXM_OPERAND_BIAS     = 3'd1;
localparam logic [2:0] VXM_OPERAND_GAMMA    = 3'd2;
localparam logic [2:0] VXM_OPERAND_BETA     = 3'd3;
localparam logic [2:0] VXM_OPERAND_ROPE_COS = 3'd4;
localparam logic [2:0] VXM_OPERAND_ROPE_SIN = 3'd5;
localparam logic [2:0] VXM_OPERAND_SCALE    = 3'd6;
localparam logic [2:0] VXM_RES_EMIT         = 3'd4;

// Encoded bus select views. Keep these as plain vectors for Icarus compatibility.
logic [2:0] westbound_consumer_sel_t;
logic [2:0] eastbound_consumer_sel_t;

logic [2:0] westbound_sel_t;
logic [2:0] eastbound_sel_t;

localparam logic [1:0] EXT_TARGET_MEM0 = 2'd0;
localparam logic [1:0] EXT_TARGET_MEM1 = 2'd1;
localparam logic [1:0] EXT_TARGET_IMEM = 2'd2;
localparam logic [1:0] EXT_TARGET_CTRL = 2'd3;

logic ext_mem0_en;
logic ext_mem1_en;
logic ext_imem_en;
logic ext_mem0_write;
logic ext_mem1_write;
logic ext_mem0_read;
logic ext_mem1_read;
logic [95:0] ext_imem_rdata;

assign ext_mem0_en    = ext_en && (ext_target == EXT_TARGET_MEM0);
assign ext_mem1_en    = ext_en && (ext_target == EXT_TARGET_MEM1);
assign ext_imem_en    = ext_en && (ext_target == EXT_TARGET_IMEM);
assign ext_mem0_write = ext_mem0_en && ext_write;
assign ext_mem1_write = ext_mem1_en && ext_write;
assign ext_mem0_read  = ext_mem0_en && !ext_write;
assign ext_mem1_read  = ext_mem1_en && !ext_write;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycle_counter <= 32'd0;
    end else if (run_en) begin
        cycle_counter <= cycle_counter + 32'd1;
    end
end

// shared bus signals (to be driven by bus fabric later)
superlane_t westbound_payload;
logic       westbound_valid;
mxm_row_t   eastbound_payload;
logic       eastbound_valid;
superlane_t eastbound_payload_lane0;

// mem0 datapath
mem_row_t   mem0_stream_in;
mem_row_t   mem0_stream_out;
logic       mem0_write_from_west;
logic       mem0_write_from_east;
logic       mem0_write_en_eff;

logic mem0_valid;
logic mem1_valid;

logic mxm_west_en;
logic sxm_west_en;
logic mem0_west_en;
logic vxm_west_en;

superlane_t sxm_stream_out_to_mxm_left;
superlane_t sxm_stream_out_to_mxm_top;

packed_fp8_row_t vxm_stream_out_live;
logic [31:0] vxm_stream_out_scale_live;
logic        vxm_out_valid_live;
packed_fp8_row_t vxm_stream_out_buf;
logic [31:0] vxm_stream_out_scale_buf;
logic        vxm_stream_out_buf_valid_w;
logic        vxm_stream_out_buf_valid_e;
logic        vxm_out_ready;
logic        vxm_result_wr_en;
logic        vxm_result_rd_en;
logic        vxm_result_full;
logic        vxm_result_empty;
logic        vxm_result_take_west;
logic        vxm_result_take_east;

// mxm datapath
logic signed [MXM_SIZE-1:0][7:0]  mxm_input_in;
logic [MXM_SIZE-1:0]              wght_load;
logic signed [MXM_SIZE-1:0][7:0]  wght_val;
logic signed [MXM_SIZE-1:0][MXM_SIZE-1:0][31:0] mxm_out;

//icu instance
icu u_icu(
    .clk(clk),
    .rst_n(rst_n),
    .run_en(run_en),
    .pc_load_en(pc_load_en),
    .pc_load_value(pc_load_value),
    .ext_imem_en(ext_imem_en),
    .ext_imem_write(ext_write),
    .ext_imem_addr(ext_addr[9:0]),
    .ext_imem_wdata(ext_wdata),
    .ext_imem_rdata(ext_imem_rdata),
    .mem0_read_en(mem0_read_en),
    .mem0_write_en(mem0_write_en),
    .mem0_addr(mem0_addr),
    .mem1_read_en(mem1_read_en),
    .mem1_write_en(mem1_write_en),
    .mem1_addr(mem1_addr),
    .sxm_opcode_input(sxm_opcode_input),
    .sxm_opcode_weight(sxm_opcode_weight),
    .sxm_load_from_west(sxm_load_from_west_ctrl),
    .vxm_ctrl(vxm_ctrl),
    .vxm_data_sel(vxm_data_sel),
    .vxm_operand_sel(vxm_operand_sel),
    .westbound_sel(westbound_sel),
    .eastbound_sel(eastbound_sel),
    .westbound_consumer_sel(westbound_consumer_sel),
    .eastbound_consumer_sel(eastbound_consumer_sel),
    .mxm_ingress_mode(mxm_ingress_mode),
    .mxm_start(mxm_start),
    .mxm_clear(mxm_clear),
    .mxm_e_row_sel(mxm_e_row_sel),
    .mxm_e_col_sel(mxm_e_col_sel),
    .mxm_e_valid_in(mxm_e_valid_in),
    .mxm_input_is_signed(mxm_input_is_signed),
    .mxm_wght_is_signed(mxm_wght_is_signed),
    .mxm_use_fp(mxm_use_fp),
    .fp_quant_mode(fp_quant_mode),
    .mem_store_fmt(mem_store_fmt),
    .vxm_rmsnorm_en(vxm_rmsnorm_en),
    .vxm_rope_en(vxm_rope_en),
    .vxm_residual_op(vxm_residual_op)
);

//mux logic for mem0 input

assign westbound_consumer_sel_t = westbound_consumer_sel;
assign eastbound_consumer_sel_t = eastbound_consumer_sel;

assign mem0_write_from_west =
    mem0_write_en && (westbound_consumer_sel_t == WC_MEM0) && westbound_valid;

assign mem0_write_from_east =
    mem0_write_en && (eastbound_consumer_sel_t == EC_MEM0) && eastbound_valid;

assign mem0_write_en_eff = ext_mem0_write || mem0_write_from_west || mem0_write_from_east;
assign mem0_read_en_eff = ext_mem0_read || mem0_read_en;
assign mem0_addr_eff = ext_mem0_en ? ext_addr[MEM_ADDR_W-1:0] : mem0_addr;

always_comb begin
    mem0_stream_in = '0;

    if (ext_mem0_write) begin
        mem0_stream_in = ext_wdata[$bits(mem_row_t)-1:0];
    end
    else if (mem0_write_from_west) begin
        if (westbound_sel_t == WB_VXM)
            mem0_stream_in = mem_row_t'(make_fp8_row_mem(westbound_payload, vxm_stream_out_scale_buf));
        else
            mem0_stream_in = mem_row_t'(make_fp8_row_mem(westbound_payload, '0));
    end
    else if (mem0_write_from_east)
        mem0_stream_in = eastbound_to_mem_row(eastbound_payload);
end

mem #(
    .DATA_W($bits(mem_row_t))
) u_mem0(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in(mem0_stream_in),
    .stream_out(mem0_stream_out),
    .read_en(mem0_read_en_eff),
    .write_en(mem0_write_en_eff),
    .addr(mem0_addr_eff),
    .ext_write_en(1'b0),
    .ext_read_en(1'b0),
    .ext_addr('0),
    .ext_data_in('0),
    .ext_data_out()
);

//since mem1 is on the far left westbound cannot write into it, so only eastbound can 
mem_row_t mem1_stream_out;

logic mem1_write_en_eff;

assign mem1_write_en_eff = ext_mem1_write ||
                            (mem1_write_en && (eastbound_consumer_sel_t == EC_MEM1) && eastbound_valid);
assign mem1_read_en_eff = ext_mem1_read || mem1_read_en;
assign mem1_addr_eff = ext_mem1_en ? ext_addr[MEM_ADDR_W-1:0] : mem1_addr;

assign westbound_sel_t = westbound_sel;

assign eastbound_sel_t = eastbound_sel;

mem #(
    .DATA_W($bits(mem_row_t))
) u_mem1(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in(ext_mem1_write ? ext_wdata[$bits(mem_row_t)-1:0] : eastbound_to_mem_row(eastbound_payload)),
    .stream_out(mem1_stream_out),
    .read_en(mem1_read_en_eff),
    .write_en(mem1_write_en_eff),
    .addr(mem1_addr_eff),
    .ext_write_en(1'b0),
    .ext_read_en(1'b0),
    .ext_addr('0),
    .ext_data_in('0),
    .ext_data_out()
);

always_comb begin
    ext_rdata = '0;
    unique case (ext_target)
        EXT_TARGET_MEM0: begin
            ext_rdata[$bits(mem_row_t)-1:0] = mem0_stream_out;
        end
        EXT_TARGET_MEM1: begin
            ext_rdata[$bits(mem_row_t)-1:0] = mem1_stream_out;
        end
        EXT_TARGET_IMEM: begin
            ext_rdata = ext_imem_rdata;
        end
        EXT_TARGET_CTRL: begin
            ext_rdata[0]     = run_en;
            ext_rdata[1]     = rst_n;
            ext_rdata[63:32] = cycle_counter;
        end
        default: begin
            ext_rdata = '0;
        end
    endcase
end

// memory read data is synchronous, so register valid alongside the read enable
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        mem0_valid <= 1'b0;
        mem1_valid <= 1'b0;
    end else begin
        mem0_valid <= mem0_read_en && !ext_mem0_read;
        mem1_valid <= mem1_read_en && !ext_mem1_read;
    end
end

logic        sxm_emit_valid;
logic        sxm_load_from_west;

westbound_bus #(
    .PAYLOAD_W($bits(superlane_t))
) u_westbound_bus(
    .producer_sel(westbound_sel_t),
    .mem0_payload(fp8_row_mem_packed(mem0_stream_out)),
    .mem1_payload(fp8_row_mem_packed(mem1_stream_out)),
    .mem0_valid(mem0_valid),
    .mem1_valid(mem1_valid),
    // SXM emits transpose rows onto westbound for downstream scheduling.
    .sxm_payload(sxm_stream_out_to_mxm_top),
    .sxm_valid(sxm_emit_valid),
    .vxm_payload(vxm_stream_out_buf),
    .vxm_valid(vxm_stream_out_buf_valid_w),
    .westbound_payload(westbound_payload),
    .westbound_valid(westbound_valid)
);

westbound_consumer_decode u_westbound_consumer_decode(
    .consumer_sel(westbound_consumer_sel_t),
    .westbound_valid(westbound_valid),
    .mxm_west_en(mxm_west_en),
    .sxm_west_en(sxm_west_en),
    .mem0_west_en(mem0_west_en),
    .vxm_west_en(vxm_west_en)
);

mxm_row_t mxm_payload_e;
logic mxm_valid_e;
mxm_row_t vxm_payload_e_bus;
mxm_row_t sxm_payload_e_bus;
mxm_row_t mem0_payload_e_bus;
mxm_row_t mem0_dequant_payload_e_bus;

assign eastbound_payload_lane0 = eastbound_payload[$bits(superlane_t)-1:0];

always_comb begin
    vxm_payload_e_bus = '0;
    sxm_payload_e_bus = '0;
    mem0_payload_e_bus = '0;

    vxm_payload_e_bus = mxm_row_t'(make_fp8_row_mem(vxm_stream_out_buf, vxm_stream_out_scale_buf));
    sxm_payload_e_bus[$bits(superlane_t)-1:0] = sxm_stream_out_to_mxm_left;
    mem0_payload_e_bus = mem_row_to_eastbound(mem0_stream_out);
end

mem_row_dequant u_mem0_dequant (
    .mem_row_i(mem0_stream_out),
    .fp32_row_o(mem0_dequant_payload_e_bus)
);

mxm_eastbound_adapter #(
    .MXM_SIZE(MXM_SIZE),
    .PAYLOAD_W(MXM_ACC_W)
) u_mxm_to_eastbound (
    .mxm_out(mxm_out),
    .mxm_row_sel(mxm_e_row_sel),
    .mxm_col_sel(mxm_e_col_sel),
    .mxm_valid_in(mxm_e_valid_in),
    .mxm_payload(mxm_payload_e),
    .mxm_valid(mxm_valid_e)
);

eastbound_bus #(
    .PAYLOAD_E($bits(mxm_row_t))
) u_eastbound_bus(
    .producer_sel(eastbound_sel_t),
    .mxm_payload_e(mxm_payload_e),
    .mxm_valid_e(mxm_valid_e),
    .vxm_payload_e(vxm_payload_e_bus),
    .vxm_valid_e(vxm_stream_out_buf_valid_e),
    .sxm_payload_e(sxm_payload_e_bus),
    .sxm_valid_e(sxm_emit_valid),
    .mem0_payload_e(mem0_payload_e_bus),
    .mem0_valid_e(mem0_valid),
    .eastbound_payload(eastbound_payload),
    .eastbound_valid(eastbound_valid)
);

logic sxm_east_en;
logic mem0_east_en;
logic vxm_east_en;
logic mem1_east_en;

eastbound_consumer_decode u_eastbound_consumer_decode(
    .consumer_sel(eastbound_consumer_sel_t),
    .eastbound_valid(eastbound_valid),
    .sxm_east_en(sxm_east_en),
    .mem0_east_en(mem0_east_en),
    .vxm_east_en(vxm_east_en),
    .mem1_east_en(mem1_east_en)
);

superlane_t sxm_e_payload_reg;
superlane_t sxm_w_payload_reg;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sxm_e_payload_reg <= '0;
        sxm_w_payload_reg <= '0;
    end else begin
        if (sxm_east_en && eastbound_valid)
            sxm_e_payload_reg <= eastbound_payload_lane0;
        if (sxm_west_en && westbound_valid)
            sxm_w_payload_reg <= westbound_payload;
    end
end

assign sxm_load_from_west = sxm_load_from_west_ctrl || (sxm_west_en && westbound_valid);

sxm #(
    .LANES(MXM_SIZE)
) u_sxm(
    .clk(clk),
    .rst_n(rst_n),
    .opcode_input(sxm_opcode_input),
    .opcode_weight(sxm_opcode_weight),
    .load_from_west(sxm_load_from_west),
    .eastbound_in(sxm_e_payload_reg),
    .westbound_in(sxm_w_payload_reg),
    .eastbound_out(sxm_stream_out_to_mxm_left),
    .westbound_out(sxm_stream_out_to_mxm_top),
    .emit_valid(sxm_emit_valid)
);

generate
    genvar i;
    for (i = 0; i < MXM_SIZE; i++) begin : g_mxm_feed
        assign mxm_input_in[i] = sxm_stream_out_to_mxm_left[i*8 +: 8];
        assign wght_val[i]     = sxm_stream_out_to_mxm_top[i*8 +: 8];
        assign wght_load[i]    = 1'b0;
    end
endgenerate

mxm #(
    .mxm_size(MXM_SIZE)
) u_mxm(
    .clk(clk),
    .rst(~rst_n),
    .mxm_clear(mxm_clear),
    .mxm_start(mxm_start),
    .westbound_payload(westbound_payload),
    .westbound_valid(westbound_valid),
    .mxm_west_en(mxm_west_en),
    .mxm_ingress_mode(mxm_ingress_mode),
    .mxm_use_fp(mxm_use_fp),
    .mxm_input_is_signed(mxm_input_is_signed),
    .mxm_wght_is_signed(mxm_wght_is_signed),
    .mxm_input_in(mxm_input_in),
    .wght_load(wght_load),
    .wght_val(wght_val),
    .mxm_out(mxm_out)
);




mxm_row_t vxm_stream_in_data;
mxm_row_t vxm_stream_in_bias;
mxm_row_t vxm_bias_reg;
logic [31:0] vxm_scale_factor_reg;
mxm_row_t vxm_rmsnorm_gamma_reg;
mxm_row_t vxm_rmsnorm_beta_reg;
superlane_t vxm_rope_cos_fp8_reg;
superlane_t vxm_rope_sin_fp8_reg;
logic     vxm_in_valid;
logic     vxm_in_ready;
logic     vxm_fifo_wr_en;
logic     vxm_fifo_rd_en;
logic     vxm_fifo_full;
logic     vxm_fifo_empty;
mxm_row_t vxm_fifo_data_out;
logic     vxm_input_overflow;
logic     vxm_residual_emit_cmd;
logic     vxm_load_operand;
logic     vxm_load_operand_east;
logic     vxm_load_operand_west;
mxm_row_t vxm_operand_payload;

assign vxm_load_operand_east = vxm_east_en && eastbound_valid;
assign vxm_load_operand_west = vxm_west_en && westbound_valid;
assign vxm_load_operand = vxm_load_operand_east || vxm_load_operand_west;

always_comb begin
    vxm_operand_payload = '0;
    if (vxm_load_operand_east) begin
        if (eastbound_sel_t == EB_MEM0) begin
            unique case (vxm_operand_sel)
                VXM_OPERAND_DATA,
                VXM_OPERAND_BIAS,
                VXM_OPERAND_GAMMA,
                VXM_OPERAND_BETA: begin
                    vxm_operand_payload = mem0_dequant_payload_e_bus;
                end
                default: begin
                    vxm_operand_payload = eastbound_payload;
                end
            endcase
        end else begin
            vxm_operand_payload = eastbound_payload;
        end
    end else if (vxm_load_operand_west) begin
        vxm_operand_payload[$bits(superlane_t)-1:0] = westbound_payload;
    end
end

// For now VXM consumes full-width rows from the eastbound bus only.
// ICU still controls whether that traffic appears on eastbound via the bus selectors.
assign vxm_fifo_wr_en = vxm_load_operand_east && (vxm_operand_sel == VXM_OPERAND_DATA);
assign vxm_residual_emit_cmd = (vxm_residual_op == VXM_RES_EMIT);
assign vxm_fifo_rd_en = !vxm_fifo_empty && vxm_in_ready && !vxm_residual_emit_cmd;
assign vxm_stream_in_data = vxm_residual_emit_cmd ? '0 : vxm_fifo_data_out;
assign vxm_stream_in_bias = vxm_bias_reg;
assign vxm_in_valid = !vxm_fifo_empty || vxm_residual_emit_cmd;

assign vxm_result_take_west = !vxm_result_empty && (westbound_sel_t == WB_VXM);
assign vxm_result_take_east = !vxm_result_empty &&
                              (westbound_sel_t != WB_VXM) &&
                              (eastbound_sel_t == EB_VXM);
assign vxm_result_rd_en = vxm_result_take_west || vxm_result_take_east;
assign vxm_stream_out_buf_valid_w = !vxm_result_empty;
assign vxm_stream_out_buf_valid_e = !vxm_result_empty && (westbound_sel_t != WB_VXM);
assign vxm_out_ready = !vxm_result_full;
assign vxm_result_wr_en = vxm_out_valid_live && vxm_out_ready;

row_fifo #(
    .DATA_W($bits(mxm_row_t)),
    .DEPTH(4)
) u_vxm_input_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(vxm_fifo_wr_en),
    .rd_en(vxm_fifo_rd_en),
    .data_in(vxm_operand_payload),
    .data_out(vxm_fifo_data_out),
    .full(vxm_fifo_full),
    .empty(vxm_fifo_empty)
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        vxm_input_overflow <= 1'b0;
        vxm_bias_reg <= '0;
        vxm_scale_factor_reg <= 32'h3f80_0000;
        vxm_rmsnorm_gamma_reg <= {MXM_SIZE{32'h3f800000}};
        vxm_rmsnorm_beta_reg <= '0;
        vxm_rope_cos_fp8_reg <= {MXM_SIZE{8'h3c}};
        vxm_rope_sin_fp8_reg <= '0;
    end else begin
        if (vxm_fifo_wr_en && vxm_fifo_full) begin
            vxm_input_overflow <= 1'b1;
        end

        if (vxm_load_operand) begin
            unique case (vxm_operand_sel)
                VXM_OPERAND_BIAS: begin
                    vxm_bias_reg <= vxm_operand_payload;
                end
                VXM_OPERAND_GAMMA: begin
                    vxm_rmsnorm_gamma_reg <= vxm_operand_payload;
                end
                VXM_OPERAND_BETA: begin
                    vxm_rmsnorm_beta_reg <= vxm_operand_payload;
                end
                VXM_OPERAND_SCALE: begin
                    vxm_scale_factor_reg <= vxm_operand_payload[31:0];
                end
                VXM_OPERAND_ROPE_COS: begin
                    vxm_rope_cos_fp8_reg <= vxm_operand_payload[$bits(superlane_t)-1:0];
                end
                VXM_OPERAND_ROPE_SIN: begin
                    vxm_rope_sin_fp8_reg <= vxm_operand_payload[$bits(superlane_t)-1:0];
                end
                default: begin
                    // Data operands are queued by u_vxm_input_fifo above.
                end
            endcase
        end
    end
end

row_fifo #(
    .DATA_W($bits(packed_fp8_row_t)),
    .DEPTH(4)
) u_vxm_output_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(vxm_result_wr_en),
    .rd_en(vxm_result_rd_en),
    .data_in(vxm_stream_out_live),
    .data_out(vxm_stream_out_buf),
    .full(vxm_result_full),
    .empty(vxm_result_empty)
);

row_fifo #(
    .DATA_W(32),
    .DEPTH(4)
) u_vxm_output_scale_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(vxm_result_wr_en),
    .rd_en(vxm_result_rd_en),
    .data_in(vxm_stream_out_scale_live),
    .data_out(vxm_stream_out_scale_buf),
    .full(),
    .empty()
);

vxm #(
    .LANES(MXM_SIZE),
    .LANE_W(32),
    .ALU_W(32),
    .RMSNORM_CHUNKS(RMSNORM_CHUNKS),
    .SOFTMAX_CHUNKS(SOFTMAX_CHUNKS)
) u_vxm(
    .clk(clk),
    .rst_n(rst_n),
    .stream_in_data(vxm_stream_in_data),
    .stream_in_bias(vxm_stream_in_bias),
    .in_valid(vxm_in_valid),
    .in_ready(vxm_in_ready),
    .vxm_ctrl(vxm_ctrl),
    .rope_en(vxm_rope_en),
    .rope_cos_fp8(vxm_rope_cos_fp8_reg),
    .rope_sin_fp8(vxm_rope_sin_fp8_reg),
    .residual_op(vxm_residual_op),
    .scale_factor(vxm_scale_factor_reg),
    .rmsnorm_bypass(~vxm_rmsnorm_en),
    .rmsnorm_gamma(vxm_rmsnorm_gamma_reg),
    .rmsnorm_beta(vxm_rmsnorm_beta_reg),
    .fp_quant_mode(fp_quant_mode),
    .stream_out(vxm_stream_out_live),
    .stream_out_scale(vxm_stream_out_scale_live),
    .out_valid(vxm_out_valid_live),
    .out_ready(vxm_out_ready)
);

endmodule
