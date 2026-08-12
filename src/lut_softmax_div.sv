`default_nettype none
`timescale 1ns/1ns

module lut_softmax_div #(
    parameter int DW = 32,
    parameter int ADDR_BITS = 8,
    parameter int RECIP_FRAC_BITS = 30
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          start,
    input  logic [DW-1:0] dividend,
    input  logic [DW-1:0] divisor,
    output logic [DW-1:0] quotient,
    output logic [DW-1:0] remainder,
    output logic          done
);

    typedef enum logic [1:0] {
        IDLE,
        LOOKUP,
        DONE_ST
    } state_t;

    typedef logic [DW-1:0] word_t;
    typedef logic [(2*DW)-1:0] product_t;

    state_t state, next_state;

    logic [DW-1:0] quotient_next;
    logic divisor_nonzero;
    logic [DW-1:0] divisor_ext;
    logic [ADDR_BITS-1:0] lut_addr;
    logic [DW-1:0] lut_value_q30;
    logic [DW-1:0] reciprocal_q30;
    product_t scaled_product;
    int unsigned msb_idx;

    function automatic int unsigned leading_one_idx(
        input logic [DW-1:0] value
    );
        int unsigned idx;
        begin
            leading_one_idx = 0;
            for (idx = 0; idx < DW; idx++) begin
                if (value[idx])
                    leading_one_idx = idx;
            end
        end
    endfunction

    // Midpoint-sampled LUT over the normalized mantissa interval [1, 2).
    // Entry i stores round((2^RECIP_FRAC_BITS) / m_i), where:
    //     m_i = 1 + (i + 0.5) / 256
    function automatic logic [DW-1:0] recip_lut(
        input logic [ADDR_BITS-1:0] addr
    );
        begin
            case (addr)
            8'h00: recip_lut = 32'h3fe00ff8;
            8'h01: recip_lut = 32'h3fa08f29;
            8'h02: recip_lut = 32'h3f618c22;
            8'h03: recip_lut = 32'h3f23056d;
            8'h04: recip_lut = 32'h3ee4f99d;
            8'h05: recip_lut = 32'h3ea76748;
            8'h06: recip_lut = 32'h3e6a4d0b;
            8'h07: recip_lut = 32'h3e2da989;
            8'h08: recip_lut = 32'h3df17b67;
            8'h09: recip_lut = 32'h3db5c153;
            8'h0a: recip_lut = 32'h3d7a79ff;
            8'h0b: recip_lut = 32'h3d3fa421;
            8'h0c: recip_lut = 32'h3d053e73;
            8'h0d: recip_lut = 32'h3ccb47b8;
            8'h0e: recip_lut = 32'h3c91beb3;
            8'h0f: recip_lut = 32'h3c58a22e;
            8'h10: recip_lut = 32'h3c1ff0f8;
            8'h11: recip_lut = 32'h3be7a9e3;
            8'h12: recip_lut = 32'h3bafcbc6;
            8'h13: recip_lut = 32'h3b78557d;
            8'h14: recip_lut = 32'h3b4145e7;
            8'h15: recip_lut = 32'h3b0a9be8;
            8'h16: recip_lut = 32'h3ad45668;
            8'h17: recip_lut = 32'h3a9e7452;
            8'h18: recip_lut = 32'h3a68f498;
            8'h19: recip_lut = 32'h3a33d62b;
            8'h1a: recip_lut = 32'h39ff1804;
            8'h1b: recip_lut = 32'h39cab91d;
            8'h1c: recip_lut = 32'h3996b877;
            8'h1d: recip_lut = 32'h39631512;
            8'h1e: recip_lut = 32'h392fcdf6;
            8'h1f: recip_lut = 32'h38fce22c;
            8'h20: recip_lut = 32'h38ca50c0;
            8'h21: recip_lut = 32'h389818c3;
            8'h22: recip_lut = 32'h38663948;
            8'h23: recip_lut = 32'h3834b166;
            8'h24: recip_lut = 32'h38038038;
            8'h25: recip_lut = 32'h37d2a4da;
            8'h26: recip_lut = 32'h37a21e6d;
            8'h27: recip_lut = 32'h3771ec13;
            8'h28: recip_lut = 32'h37420cf3;
            8'h29: recip_lut = 32'h37128037;
            8'h2a: recip_lut = 32'h36e3450a;
            8'h2b: recip_lut = 32'h36b45a9b;
            8'h2c: recip_lut = 32'h3685c01b;
            8'h2d: recip_lut = 32'h365774c0;
            8'h2e: recip_lut = 32'h362977c0;
            8'h2f: recip_lut = 32'h35fbc854;
            8'h30: recip_lut = 32'h35ce65ba;
            8'h31: recip_lut = 32'h35a14f30;
            8'h32: recip_lut = 32'h357483f8;
            8'h33: recip_lut = 32'h35480355;
            8'h34: recip_lut = 32'h351bcc8d;
            8'h35: recip_lut = 32'h34efdeea;
            8'h36: recip_lut = 32'h34c439b7;
            8'h37: recip_lut = 32'h3498dc40;
            8'h38: recip_lut = 32'h346dc5d6;
            8'h39: recip_lut = 32'h3442f5cb;
            8'h3a: recip_lut = 32'h34186b72;
            8'h3b: recip_lut = 32'h33ee2623;
            8'h3c: recip_lut = 32'h33c42535;
            8'h3d: recip_lut = 32'h339a6803;
            8'h3e: recip_lut = 32'h3370edea;
            8'h3f: recip_lut = 32'h3347b649;
            8'h40: recip_lut = 32'h331ec080;
            8'h41: recip_lut = 32'h32f60bf2;
            8'h42: recip_lut = 32'h32cd9803;
            8'h43: recip_lut = 32'h32a5641b;
            8'h44: recip_lut = 32'h327d6fa1;
            8'h45: recip_lut = 32'h3255ba01;
            8'h46: recip_lut = 32'h322e42a5;
            8'h47: recip_lut = 32'h320708fd;
            8'h48: recip_lut = 32'h31e00c78;
            8'h49: recip_lut = 32'h31b94c87;
            8'h4a: recip_lut = 32'h3192c89e;
            8'h4b: recip_lut = 32'h316c8031;
            8'h4c: recip_lut = 32'h314672b8;
            8'h4d: recip_lut = 32'h31209faa;
            8'h4e: recip_lut = 32'h30fb0681;
            8'h4f: recip_lut = 32'h30d5a6b9;
            8'h50: recip_lut = 32'h30b07fcf;
            8'h51: recip_lut = 32'h308b9142;
            8'h52: recip_lut = 32'h3066da90;
            8'h53: recip_lut = 32'h30425b3d;
            8'h54: recip_lut = 32'h301e12cc;
            8'h55: recip_lut = 32'h2ffa00c0;
            8'h56: recip_lut = 32'h2fd624a0;
            8'h57: recip_lut = 32'h2fb27df3;
            8'h58: recip_lut = 32'h2f8f0c43;
            8'h59: recip_lut = 32'h2f6bcf19;
            8'h5a: recip_lut = 32'h2f48c601;
            8'h5b: recip_lut = 32'h2f25f088;
            8'h5c: recip_lut = 32'h2f034e3c;
            8'h5d: recip_lut = 32'h2ee0deac;
            8'h5e: recip_lut = 32'h2ebea16a;
            8'h5f: recip_lut = 32'h2e9c9608;
            8'h60: recip_lut = 32'h2e7abc19;
            8'h61: recip_lut = 32'h2e591331;
            8'h62: recip_lut = 32'h2e379ae6;
            8'h63: recip_lut = 32'h2e1652d0;
            8'h64: recip_lut = 32'h2df53a86;
            8'h65: recip_lut = 32'h2dd451a2;
            8'h66: recip_lut = 32'h2db397be;
            8'h67: recip_lut = 32'h2d930c76;
            8'h68: recip_lut = 32'h2d72af67;
            8'h69: recip_lut = 32'h2d52802d;
            8'h6a: recip_lut = 32'h2d327e69;
            8'h6b: recip_lut = 32'h2d12a9ba;
            8'h6c: recip_lut = 32'h2cf301c1;
            8'h6d: recip_lut = 32'h2cd38621;
            8'h6e: recip_lut = 32'h2cb4367c;
            8'h6f: recip_lut = 32'h2c951276;
            8'h70: recip_lut = 32'h2c7619b4;
            8'h71: recip_lut = 32'h2c574bdd;
            8'h72: recip_lut = 32'h2c38a898;
            8'h73: recip_lut = 32'h2c1a2f8c;
            8'h74: recip_lut = 32'h2bfbe063;
            8'h75: recip_lut = 32'h2bddbac6;
            8'h76: recip_lut = 32'h2bbfbe60;
            8'h77: recip_lut = 32'h2ba1eade;
            8'h78: recip_lut = 32'h2b843fea;
            8'h79: recip_lut = 32'h2b66bd34;
            8'h7a: recip_lut = 32'h2b496269;
            8'h7b: recip_lut = 32'h2b2c2f38;
            8'h7c: recip_lut = 32'h2b0f2352;
            8'h7d: recip_lut = 32'h2af23e68;
            8'h7e: recip_lut = 32'h2ad5802b;
            8'h7f: recip_lut = 32'h2ab8e84d;
            8'h80: recip_lut = 32'h2a9c7683;
            8'h81: recip_lut = 32'h2a802a80;
            8'h82: recip_lut = 32'h2a6403f9;
            8'h83: recip_lut = 32'h2a4802a5;
            8'h84: recip_lut = 32'h2a2c2638;
            8'h85: recip_lut = 32'h2a106e6b;
            8'h86: recip_lut = 32'h29f4daf6;
            8'h87: recip_lut = 32'h29d96b91;
            8'h88: recip_lut = 32'h29be1ff6;
            8'h89: recip_lut = 32'h29a2f7de;
            8'h8a: recip_lut = 32'h2987f306;
            8'h8b: recip_lut = 32'h296d1127;
            8'h8c: recip_lut = 32'h295251ff;
            8'h8d: recip_lut = 32'h2937b54b;
            8'h8e: recip_lut = 32'h291d3ac8;
            8'h8f: recip_lut = 32'h2902e234;
            8'h90: recip_lut = 32'h28e8ab4e;
            8'h91: recip_lut = 32'h28ce95d7;
            8'h92: recip_lut = 32'h28b4a18d;
            8'h93: recip_lut = 32'h289ace32;
            8'h94: recip_lut = 32'h28811b88;
            8'h95: recip_lut = 32'h28678950;
            8'h96: recip_lut = 32'h284e174d;
            8'h97: recip_lut = 32'h2834c543;
            8'h98: recip_lut = 32'h281b92f5;
            8'h99: recip_lut = 32'h28028028;
            8'h9a: recip_lut = 32'h27e98ca1;
            8'h9b: recip_lut = 32'h27d0b825;
            8'h9c: recip_lut = 32'h27b8027c;
            8'h9d: recip_lut = 32'h279f6b6a;
            8'h9e: recip_lut = 32'h2786f2b9;
            8'h9f: recip_lut = 32'h276e982f;
            8'ha0: recip_lut = 32'h27565b95;
            8'ha1: recip_lut = 32'h273e3cb4;
            8'ha2: recip_lut = 32'h27263b56;
            8'ha3: recip_lut = 32'h270e5744;
            8'ha4: recip_lut = 32'h26f69049;
            8'ha5: recip_lut = 32'h26dee630;
            8'ha6: recip_lut = 32'h26c758c4;
            8'ha7: recip_lut = 32'h26afe7d2;
            8'ha8: recip_lut = 32'h26989326;
            8'ha9: recip_lut = 32'h26815a8c;
            8'haa: recip_lut = 32'h266a3dd3;
            8'hab: recip_lut = 32'h26533cc8;
            8'hac: recip_lut = 32'h263c573a;
            8'had: recip_lut = 32'h26258cf7;
            8'hae: recip_lut = 32'h260eddcf;
            8'haf: recip_lut = 32'h25f84991;
            8'hb0: recip_lut = 32'h25e1d00e;
            8'hb1: recip_lut = 32'h25cb7117;
            8'hb2: recip_lut = 32'h25b52c7c;
            8'hb3: recip_lut = 32'h259f020f;
            8'hb4: recip_lut = 32'h2588f1a2;
            8'hb5: recip_lut = 32'h2572fb07;
            8'hb6: recip_lut = 32'h255d1e11;
            8'hb7: recip_lut = 32'h25475a93;
            8'hb8: recip_lut = 32'h2531b062;
            8'hb9: recip_lut = 32'h251c1f50;
            8'hba: recip_lut = 32'h2506a732;
            8'hbb: recip_lut = 32'h24f147dd;
            8'hbc: recip_lut = 32'h24dc0127;
            8'hbd: recip_lut = 32'h24c6d2e4;
            8'hbe: recip_lut = 32'h24b1bceb;
            8'hbf: recip_lut = 32'h249cbf12;
            8'hc0: recip_lut = 32'h2487d930;
            8'hc1: recip_lut = 32'h24730b1b;
            8'hc2: recip_lut = 32'h245e54ac;
            8'hc3: recip_lut = 32'h2449b5b9;
            8'hc4: recip_lut = 32'h24352e1c;
            8'hc5: recip_lut = 32'h2420bdac;
            8'hc6: recip_lut = 32'h240c6442;
            8'hc7: recip_lut = 32'h23f821b9;
            8'hc8: recip_lut = 32'h23e3f5e8;
            8'hc9: recip_lut = 32'h23cfe0aa;
            8'hca: recip_lut = 32'h23bbe1d9;
            8'hcb: recip_lut = 32'h23a7f951;
            8'hcc: recip_lut = 32'h239426ea;
            8'hcd: recip_lut = 32'h23806a81;
            8'hce: recip_lut = 32'h236cc3f2;
            8'hcf: recip_lut = 32'h23593317;
            8'hd0: recip_lut = 32'h2345b7cd;
            8'hd1: recip_lut = 32'h233251f1;
            8'hd2: recip_lut = 32'h231f015f;
            8'hd3: recip_lut = 32'h230bc5f5;
            8'hd4: recip_lut = 32'h22f89f8e;
            8'hd5: recip_lut = 32'h22e58e0a;
            8'hd6: recip_lut = 32'h22d29146;
            8'hd7: recip_lut = 32'h22bfa921;
            8'hd8: recip_lut = 32'h22acd578;
            8'hd9: recip_lut = 32'h229a162b;
            8'hda: recip_lut = 32'h22876b18;
            8'hdb: recip_lut = 32'h2274d41f;
            8'hdc: recip_lut = 32'h22625120;
            8'hdd: recip_lut = 32'h224fe1fa;
            8'hde: recip_lut = 32'h223d868e;
            8'hdf: recip_lut = 32'h222b3ebb;
            8'he0: recip_lut = 32'h22190a64;
            8'he1: recip_lut = 32'h2206e967;
            8'he2: recip_lut = 32'h21f4dba8;
            8'he3: recip_lut = 32'h21e2e107;
            8'he4: recip_lut = 32'h21d0f965;
            8'he5: recip_lut = 32'h21bf24a6;
            8'he6: recip_lut = 32'h21ad62aa;
            8'he7: recip_lut = 32'h219bb355;
            8'he8: recip_lut = 32'h218a1689;
            8'he9: recip_lut = 32'h21788c29;
            8'hea: recip_lut = 32'h21671418;
            8'heb: recip_lut = 32'h2155ae3a;
            8'hec: recip_lut = 32'h21445a72;
            8'hed: recip_lut = 32'h213318a4;
            8'hee: recip_lut = 32'h2121e8b4;
            8'hef: recip_lut = 32'h2110ca87;
            8'hf0: recip_lut = 32'h20ffbe01;
            8'hf1: recip_lut = 32'h20eec306;
            8'hf2: recip_lut = 32'h20ddd97c;
            8'hf3: recip_lut = 32'h20cd0148;
            8'hf4: recip_lut = 32'h20bc3a4f;
            8'hf5: recip_lut = 32'h20ab8477;
            8'hf6: recip_lut = 32'h209adfa6;
            8'hf7: recip_lut = 32'h208a4bc2;
            8'hf8: recip_lut = 32'h2079c8b1;
            8'hf9: recip_lut = 32'h20695659;
            8'hfa: recip_lut = 32'h2058f4a1;
            8'hfb: recip_lut = 32'h2048a370;
            8'hfc: recip_lut = 32'h203862ad;
            8'hfd: recip_lut = 32'h2028323f;
            8'hfe: recip_lut = 32'h2018120e;
            8'hff: recip_lut = 32'h20080201;
            default: recip_lut = '0;
            endcase
        end
    endfunction

    always_comb begin
        divisor_nonzero = (divisor != '0);
        msb_idx = leading_one_idx(divisor);
        divisor_ext = divisor;
        lut_addr = '0;
        lut_value_q30 = '0;
        reciprocal_q30 = '0;
        scaled_product = '0;
        quotient_next = '0;

        if (divisor_nonzero) begin
            for (int frac_idx = 0; frac_idx < ADDR_BITS; frac_idx++) begin
                if (msb_idx > frac_idx)
                    lut_addr[ADDR_BITS-1-frac_idx] = divisor_ext[msb_idx-1-frac_idx];
            end

            lut_value_q30 = recip_lut(lut_addr);
            reciprocal_q30 = lut_value_q30 >> msb_idx;
            scaled_product = product_t'({{DW{1'b0}}, dividend}) *
                             product_t'({{DW{1'b0}}, reciprocal_q30});
            quotient_next = word_t'(scaled_product >> RECIP_FRAC_BITS);
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            quotient <= '0;
            remainder <= '0;
        end else begin
            state <= next_state;

            if (state == LOOKUP) begin
                quotient <= quotient_next;
                remainder <= '0;
            end
        end
    end

    always_comb begin
        next_state = state;
        done = 1'b0;

        case (state)
            IDLE: begin
                if (start)
                    next_state = LOOKUP;
            end

            LOOKUP: begin
                next_state = DONE_ST;
            end

            DONE_ST: begin
                done = 1'b1;
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

endmodule

`default_nettype wire
