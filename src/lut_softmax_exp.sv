// ==============================================================================
// LUT-based exponential approximation for the softmax module.
//
// The block approximates:
//     q_out ~= exp(q)
//
// by using range reduction: q = -z * LN2 + p where p in [-LN2, 0].
// Since p is in [-177, 0], we use a 178-entry lookup table.
// ==============================================================================
`default_nettype none
`timescale 1ns/1ns

module lut_softmax_exp #(
    parameter int DW = 32
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic signed [DW-1:0] q,
    output logic signed [DW-1:0] q_out
);

    localparam int LN2 = 177;      // ln(2) * 256 ≈ 177

    // Decompose q into z and p: q = -z * LN2 + p
    logic signed [DW-1:0] z;
    assign z = -q / LN2;

    logic signed [DW-1:0] p;
    assign p = q + (z * LN2);

    // Address is the absolute value of the remainder p: addr = -p
    // Since p is in [-177, 0], addr is in [0, 177] (which fits in 8 bits)
    logic [7:0] lut_addr;
    assign lut_addr = 8'(-p);

    // 178-entry lookup table for exp(p) scaled by 256
    function automatic logic [DW-1:0] exp_lut(
        input logic [7:0] addr
    );
        begin
            case (addr)
            8'h00: exp_lut = 32'd256;
            8'h01: exp_lut = 32'd255;
            8'h02: exp_lut = 32'd254;
            8'h03: exp_lut = 32'd253;
            8'h04: exp_lut = 32'd252;
            8'h05: exp_lut = 32'd251;
            8'h06: exp_lut = 32'd250;
            8'h07: exp_lut = 32'd249;
            8'h08: exp_lut = 32'd248;
            8'h09: exp_lut = 32'd247;
            8'h0a: exp_lut = 32'd246;
            8'h0b: exp_lut = 32'd245;
            8'h0c: exp_lut = 32'd244;
            8'h0d: exp_lut = 32'd243;
            8'h0e: exp_lut = 32'd242;
            8'h0f: exp_lut = 32'd241;
            8'h10: exp_lut = 32'd240;
            8'h11: exp_lut = 32'd240;
            8'h12: exp_lut = 32'd239;
            8'h13: exp_lut = 32'd238;
            8'h14: exp_lut = 32'd237;
            8'h15: exp_lut = 32'd236;
            8'h16: exp_lut = 32'd235;
            8'h17: exp_lut = 32'd234;
            8'h18: exp_lut = 32'd233;
            8'h19: exp_lut = 32'd232;
            8'h1a: exp_lut = 32'd231;
            8'h1b: exp_lut = 32'd230;
            8'h1c: exp_lut = 32'd229;
            8'h1d: exp_lut = 32'd229;
            8'h1e: exp_lut = 32'd228;
            8'h1f: exp_lut = 32'd227;
            8'h20: exp_lut = 32'd226;
            8'h21: exp_lut = 32'd225;
            8'h22: exp_lut = 32'd224;
            8'h23: exp_lut = 32'd223;
            8'h24: exp_lut = 32'd222;
            8'h25: exp_lut = 32'd222;
            8'h26: exp_lut = 32'd221;
            8'h27: exp_lut = 32'd220;
            8'h28: exp_lut = 32'd219;
            8'h29: exp_lut = 32'd218;
            8'h2a: exp_lut = 32'd217;
            8'h2b: exp_lut = 32'd216;
            8'h2c: exp_lut = 32'd216;
            8'h2d: exp_lut = 32'd215;
            8'h2e: exp_lut = 32'd214;
            8'h2f: exp_lut = 32'd213;
            8'h30: exp_lut = 32'd212;
            8'h31: exp_lut = 32'd211;
            8'h32: exp_lut = 32'd211;
            8'h33: exp_lut = 32'd210;
            8'h34: exp_lut = 32'd209;
            8'h35: exp_lut = 32'd208;
            8'h36: exp_lut = 32'd207;
            8'h37: exp_lut = 32'd207;
            8'h38: exp_lut = 32'd206;
            8'h39: exp_lut = 32'd205;
            8'h3a: exp_lut = 32'd204;
            8'h3b: exp_lut = 32'd203;
            8'h3c: exp_lut = 32'd203;
            8'h3d: exp_lut = 32'd202;
            8'h3e: exp_lut = 32'd201;
            8'h3f: exp_lut = 32'd200;
            8'h40: exp_lut = 32'd199;
            8'h41: exp_lut = 32'd199;
            8'h42: exp_lut = 32'd198;
            8'h43: exp_lut = 32'd197;
            8'h44: exp_lut = 32'd196;
            8'h45: exp_lut = 32'd196;
            8'h46: exp_lut = 32'd195;
            8'h47: exp_lut = 32'd194;
            8'h48: exp_lut = 32'd193;
            8'h49: exp_lut = 32'd192;
            8'h4a: exp_lut = 32'd192;
            8'h4b: exp_lut = 32'd191;
            8'h4c: exp_lut = 32'd190;
            8'h4d: exp_lut = 32'd190;
            8'h4e: exp_lut = 32'd189;
            8'h4f: exp_lut = 32'd188;
            8'h50: exp_lut = 32'd187;
            8'h51: exp_lut = 32'd187;
            8'h52: exp_lut = 32'd186;
            8'h53: exp_lut = 32'd185;
            8'h54: exp_lut = 32'd184;
            8'h55: exp_lut = 32'd184;
            8'h56: exp_lut = 32'd183;
            8'h57: exp_lut = 32'd182;
            8'h58: exp_lut = 32'd182;
            8'h59: exp_lut = 32'd181;
            8'h5a: exp_lut = 32'd180;
            8'h5b: exp_lut = 32'd179;
            8'h5c: exp_lut = 32'd179;
            8'h5d: exp_lut = 32'd178;
            8'h5e: exp_lut = 32'd177;
            8'h5f: exp_lut = 32'd177;
            8'h60: exp_lut = 32'd176;
            8'h61: exp_lut = 32'd175;
            8'h62: exp_lut = 32'd175;
            8'h63: exp_lut = 32'd174;
            8'h64: exp_lut = 32'd173;
            8'h65: exp_lut = 32'd173;
            8'h66: exp_lut = 32'd172;
            8'h67: exp_lut = 32'd171;
            8'h68: exp_lut = 32'd171;
            8'h69: exp_lut = 32'd170;
            8'h6a: exp_lut = 32'd169;
            8'h6b: exp_lut = 32'd169;
            8'h6c: exp_lut = 32'd168;
            8'h6d: exp_lut = 32'd167;
            8'h6e: exp_lut = 32'd167;
            8'h6f: exp_lut = 32'd166;
            8'h70: exp_lut = 32'd165;
            8'h71: exp_lut = 32'd165;
            8'h72: exp_lut = 32'd164;
            8'h73: exp_lut = 32'd163;
            8'h74: exp_lut = 32'd163;
            8'h75: exp_lut = 32'd162;
            8'h76: exp_lut = 32'd161;
            8'h77: exp_lut = 32'd161;
            8'h78: exp_lut = 32'd160;
            8'h79: exp_lut = 32'd160;
            8'h7a: exp_lut = 32'd159;
            8'h7b: exp_lut = 32'd158;
            8'h7c: exp_lut = 32'd158;
            8'h7d: exp_lut = 32'd157;
            8'h7e: exp_lut = 32'd156;
            8'h7f: exp_lut = 32'd156;
            8'h80: exp_lut = 32'd155;
            8'h81: exp_lut = 32'd155;
            8'h82: exp_lut = 32'd154;
            8'h83: exp_lut = 32'd153;
            8'h84: exp_lut = 32'd153;
            8'h85: exp_lut = 32'd152;
            8'h86: exp_lut = 32'd152;
            8'h87: exp_lut = 32'd151;
            8'h88: exp_lut = 32'd150;
            8'h89: exp_lut = 32'd150;
            8'h8a: exp_lut = 32'd149;
            8'h8b: exp_lut = 32'd149;
            8'h8c: exp_lut = 32'd148;
            8'h8d: exp_lut = 32'd148;
            8'h8e: exp_lut = 32'd147;
            8'h8f: exp_lut = 32'd146;
            8'h90: exp_lut = 32'd146;
            8'h91: exp_lut = 32'd145;
            8'h92: exp_lut = 32'd145;
            8'h93: exp_lut = 32'd144;
            8'h94: exp_lut = 32'd144;
            8'h95: exp_lut = 32'd143;
            8'h96: exp_lut = 32'd142;
            8'h97: exp_lut = 32'd142;
            8'h98: exp_lut = 32'd141;
            8'h99: exp_lut = 32'd141;
            8'h9a: exp_lut = 32'd140;
            8'h9b: exp_lut = 32'd140;
            8'h9c: exp_lut = 32'd139;
            8'h9d: exp_lut = 32'd139;
            8'h9e: exp_lut = 32'd138;
            8'h9f: exp_lut = 32'd138;
            8'ha0: exp_lut = 32'd137;
            8'ha1: exp_lut = 32'd136;
            8'ha2: exp_lut = 32'd136;
            8'ha3: exp_lut = 32'd135;
            8'ha4: exp_lut = 32'd135;
            8'ha5: exp_lut = 32'd134;
            8'ha6: exp_lut = 32'd134;
            8'ha7: exp_lut = 32'd133;
            8'ha8: exp_lut = 32'd133;
            8'ha9: exp_lut = 32'd132;
            8'haa: exp_lut = 32'd132;
            8'hab: exp_lut = 32'd131;
            8'hac: exp_lut = 32'd131;
            8'had: exp_lut = 32'd130;
            8'hae: exp_lut = 32'd130;
            8'haf: exp_lut = 32'd129;
            8'hb0: exp_lut = 32'd129;
            8'hb1: exp_lut = 32'd128;
            default: exp_lut = 32'd0;
            endcase
        end
    endfunction

    // Lookup and scale
    logic [DW-1:0] lut_value;
    assign lut_value = exp_lut(lut_addr);

    // Shift by the integer quotient part
    assign q_out = lut_value >>> z;

endmodule
