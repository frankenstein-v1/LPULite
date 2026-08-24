`timescale 1ns/1ps

// Precomputed RoPE trigonometric ROM for the 8-wide RoPE datapath used by the
// repository's RoPE-enabled tiny-language-model flows. The shipped MicroGPT
// checkpoint uses learned position embeddings and does not enable this path.
// Each address is a token position. Each 32-bit output contains four signed
// Q1.7 coefficients, one for each adjacent lane pair. Values were generated as
// round(127*cos/sin(position / 10000^(2*pair/8))) and saturate to [-127,127].
// Keeping this ROM combinational avoids adding a cycle to the existing RoPE
// handshake; FPGA synthesis may implement it as distributed ROM/logic.
module rope_trig_lut (
    input  logic [7:0]  position_i,
    output logic [31:0] cos_pairs_q1_7_o,
    output logic [31:0] sin_pairs_q1_7_o
);
    always_comb begin
        unique case (position_i)
            8'd0: begin cos_pairs_q1_7_o = 32'h7f7f7f7f; sin_pairs_q1_7_o = 32'h00000000; end
            8'd1: begin cos_pairs_q1_7_o = 32'h7f7f7e45; sin_pairs_q1_7_o = 32'h00010d6b; end
            8'd2: begin cos_pairs_q1_7_o = 32'h7f7f7ccb; sin_pairs_q1_7_o = 32'h00031973; end
            8'd3: begin cos_pairs_q1_7_o = 32'h7f7f7982; sin_pairs_q1_7_o = 32'h00042612; end
            8'd4: begin cos_pairs_q1_7_o = 32'h7f7f75ad; sin_pairs_q1_7_o = 32'h010531a0; end
            8'd5: begin cos_pairs_q1_7_o = 32'h7f7f6f24; sin_pairs_q1_7_o = 32'h01063d86; end
            8'd6: begin cos_pairs_q1_7_o = 32'h7f7f697a; sin_pairs_q1_7_o = 32'h010848dd; end
            8'd7: begin cos_pairs_q1_7_o = 32'h7f7f6160; sin_pairs_q1_7_o = 32'h01095253; end
            8'd8: begin cos_pairs_q1_7_o = 32'h7f7f58ee; sin_pairs_q1_7_o = 32'h010a5b7e; end
            8'd9: begin cos_pairs_q1_7_o = 32'h7f7e4f8c; sin_pairs_q1_7_o = 32'h010b6334; end
            8'd10: begin cos_pairs_q1_7_o = 32'h7f7e4595; sin_pairs_q1_7_o = 32'h010d6bbb; end
            8'd11: begin cos_pairs_q1_7_o = 32'h7f7e3a01; sin_pairs_q1_7_o = 32'h010e7181; end
            8'd12: begin cos_pairs_q1_7_o = 32'h7f7e2e6b; sin_pairs_q1_7_o = 32'h020f76bc; end
            8'd13: begin cos_pairs_q1_7_o = 32'h7f7e2273; sin_pairs_q1_7_o = 32'h02107a35; end
            8'd14: begin cos_pairs_q1_7_o = 32'h7f7e1611; sin_pairs_q1_7_o = 32'h02127d7e; end
            8'd15: begin cos_pairs_q1_7_o = 32'h7f7e09a0; sin_pairs_q1_7_o = 32'h02137f53; end
            8'd16: begin cos_pairs_q1_7_o = 32'h7f7dfc86; sin_pairs_q1_7_o = 32'h02147fdb; end
            8'd17: begin cos_pairs_q1_7_o = 32'h7f7df0dd; sin_pairs_q1_7_o = 32'h02157e86; end
            8'd18: begin cos_pairs_q1_7_o = 32'h7f7de354; sin_pairs_q1_7_o = 32'h02177ca1; end
            8'd19: begin cos_pairs_q1_7_o = 32'h7f7dd77e; sin_pairs_q1_7_o = 32'h02187813; end
            8'd20: begin cos_pairs_q1_7_o = 32'h7f7ccb34; sin_pairs_q1_7_o = 32'h03197374; end
            8'd21: begin cos_pairs_q1_7_o = 32'h7f7cc0ba; sin_pairs_q1_7_o = 32'h031a6e6a; end
            8'd22: begin cos_pairs_q1_7_o = 32'h7f7cb581; sin_pairs_q1_7_o = 32'h031c67ff; end
            8'd23: begin cos_pairs_q1_7_o = 32'h7f7cabbc; sin_pairs_q1_7_o = 32'h031d5f95; end
            8'd24: begin cos_pairs_q1_7_o = 32'h7f7ba236; sin_pairs_q1_7_o = 32'h031e568d; end
            8'd25: begin cos_pairs_q1_7_o = 32'h7f7b9a7e; sin_pairs_q1_7_o = 32'h031f4cef; end
            8'd26: begin cos_pairs_q1_7_o = 32'h7f7b9352; sin_pairs_q1_7_o = 32'h03214161; end
            8'd27: begin cos_pairs_q1_7_o = 32'h7f7a8ddb; sin_pairs_q1_7_o = 32'h03223679; end
            8'd28: begin cos_pairs_q1_7_o = 32'h7f7a8886; sin_pairs_q1_7_o = 32'h04232b22; end
            8'd29: begin cos_pairs_q1_7_o = 32'h7f7a85a1; sin_pairs_q1_7_o = 32'h04241eac; end
            8'd30: begin cos_pairs_q1_7_o = 32'h7f798214; sin_pairs_q1_7_o = 32'h04261283; end
            8'd31: begin cos_pairs_q1_7_o = 32'h7f798174; sin_pairs_q1_7_o = 32'h042705cd; end
            8'd32: begin cos_pairs_q1_7_o = 32'h7f79816a; sin_pairs_q1_7_o = 32'h0428f946; end
            8'd33: begin cos_pairs_q1_7_o = 32'h7f7883fe; sin_pairs_q1_7_o = 32'h0429ec7f; end
            8'd34: begin cos_pairs_q1_7_o = 32'h7f788594; sin_pairs_q1_7_o = 32'h042ae043; end
            8'd35: begin cos_pairs_q1_7_o = 32'h7f77898d; sin_pairs_q1_7_o = 32'h042cd3ca; end
            8'd36: begin cos_pairs_q1_7_o = 32'h7f778ef0; sin_pairs_q1_7_o = 32'h052dc882; end
            8'd37: begin cos_pairs_q1_7_o = 32'h7f769461; sin_pairs_q1_7_o = 32'h052ebdae; end
            8'd38: begin cos_pairs_q1_7_o = 32'h7f769c79; sin_pairs_q1_7_o = 32'h052fb226; end
            8'd39: begin cos_pairs_q1_7_o = 32'h7f75a422; sin_pairs_q1_7_o = 32'h0530a97a; end
            8'd40: begin cos_pairs_q1_7_o = 32'h7f75adab; sin_pairs_q1_7_o = 32'h0531a05f; end
            8'd41: begin cos_pairs_q1_7_o = 32'h7f74b783; sin_pairs_q1_7_o = 32'h053398ec; end
            8'd42: begin cos_pairs_q1_7_o = 32'h7f74c2cd; sin_pairs_q1_7_o = 32'h0534918c; end
            8'd43: begin cos_pairs_q1_7_o = 32'h7f73cd46; sin_pairs_q1_7_o = 32'h05358c96; end
            8'd44: begin cos_pairs_q1_7_o = 32'h7f73d97f; sin_pairs_q1_7_o = 32'h06368702; end
            8'd45: begin cos_pairs_q1_7_o = 32'h7f72e543; sin_pairs_q1_7_o = 32'h0637846c; end
            8'd46: begin cos_pairs_q1_7_o = 32'h7f72f2c9; sin_pairs_q1_7_o = 32'h06388273; end
            8'd47: begin cos_pairs_q1_7_o = 32'h7f71fe82; sin_pairs_q1_7_o = 32'h063a8110; end
            8'd48: begin cos_pairs_q1_7_o = 32'h7f710baf; sin_pairs_q1_7_o = 32'h063b819e; end
            8'd49: begin cos_pairs_q1_7_o = 32'h7f701826; sin_pairs_q1_7_o = 32'h063c8387; end
            8'd50: begin cos_pairs_q1_7_o = 32'h7f6f247b; sin_pairs_q1_7_o = 32'h063d86df; end
            8'd51: begin cos_pairs_q1_7_o = 32'h7f6f305e; sin_pairs_q1_7_o = 32'h063e8a55; end
            8'd52: begin cos_pairs_q1_7_o = 32'h7f6e3ceb; sin_pairs_q1_7_o = 32'h073f907d; end
            8'd53: begin cos_pairs_q1_7_o = 32'h7f6e468b; sin_pairs_q1_7_o = 32'h07409632; end
            8'd54: begin cos_pairs_q1_7_o = 32'h7f6d5197; sin_pairs_q1_7_o = 32'h07419eb9; end
            8'd55: begin cos_pairs_q1_7_o = 32'h7f6c5a03; sin_pairs_q1_7_o = 32'h0742a681; end
            8'd56: begin cos_pairs_q1_7_o = 32'h7f6c626c; sin_pairs_q1_7_o = 32'h0743b0be; end
            8'd57: begin cos_pairs_q1_7_o = 32'h7f6b6a72; sin_pairs_q1_7_o = 32'h0745ba37; end
            8'd58: begin cos_pairs_q1_7_o = 32'h7f6a700f; sin_pairs_q1_7_o = 32'h0746c57e; end
            8'd59: begin cos_pairs_q1_7_o = 32'h7f6a769e; sin_pairs_q1_7_o = 32'h0747d151; end
            8'd60: begin cos_pairs_q1_7_o = 32'h7f697a87; sin_pairs_q1_7_o = 32'h0848ddd9; end
            8'd61: begin cos_pairs_q1_7_o = 32'h7f687ddf; sin_pairs_q1_7_o = 32'h0849e985; end
            8'd62: begin cos_pairs_q1_7_o = 32'h7f677f56; sin_pairs_q1_7_o = 32'h084af5a2; end
            8'd63: begin cos_pairs_q1_7_o = 32'h7f677f7d; sin_pairs_q1_7_o = 32'h084b0215; end
            8'd64: begin cos_pairs_q1_7_o = 32'h7f667e32; sin_pairs_q1_7_o = 32'h084c0f75; end
            8'd65: begin cos_pairs_q1_7_o = 32'h7f657cb9; sin_pairs_q1_7_o = 32'h084d1b69; end
            8'd66: begin cos_pairs_q1_7_o = 32'h7f647981; sin_pairs_q1_7_o = 32'h084e28fd; end
            8'd67: begin cos_pairs_q1_7_o = 32'h7f6474be; sin_pairs_q1_7_o = 32'h094f3393; end
            8'd68: begin cos_pairs_q1_7_o = 32'h7f636e38; sin_pairs_q1_7_o = 32'h09503f8e; end
            8'd69: begin cos_pairs_q1_7_o = 32'h7f62687e; sin_pairs_q1_7_o = 32'h095149f1; end
            8'd70: begin cos_pairs_q1_7_o = 32'h7f616050; sin_pairs_q1_7_o = 32'h09525362; end
            8'd71: begin cos_pairs_q1_7_o = 32'h7f6057d9; sin_pairs_q1_7_o = 32'h09535d79; end
            8'd72: begin cos_pairs_q1_7_o = 32'h7f5f4d85; sin_pairs_q1_7_o = 32'h09546520; end
            8'd73: begin cos_pairs_q1_7_o = 32'h7f5f43a3; sin_pairs_q1_7_o = 32'h09556caa; end
            8'd74: begin cos_pairs_q1_7_o = 32'h7f5e3816; sin_pairs_q1_7_o = 32'h09567283; end
            8'd75: begin cos_pairs_q1_7_o = 32'h7f5d2c75; sin_pairs_q1_7_o = 32'h0a5777cf; end
            8'd76: begin cos_pairs_q1_7_o = 32'h7f5c2069; sin_pairs_q1_7_o = 32'h0a577b48; end
            8'd77: begin cos_pairs_q1_7_o = 32'h7f5b13fc; sin_pairs_q1_7_o = 32'h0a587d7f; end
            8'd78: begin cos_pairs_q1_7_o = 32'h7f5a0793; sin_pairs_q1_7_o = 32'h0a597f41; end
            8'd79: begin cos_pairs_q1_7_o = 32'h7f59fa8e; sin_pairs_q1_7_o = 32'h0a5a7fc8; end
            8'd80: begin cos_pairs_q1_7_o = 32'h7f58eef2; sin_pairs_q1_7_o = 32'h0a5b7e82; end
            8'd81: begin cos_pairs_q1_7_o = 32'h7f58e163; sin_pairs_q1_7_o = 32'h0a5c7bb0; end
            8'd82: begin cos_pairs_q1_7_o = 32'h7f57d579; sin_pairs_q1_7_o = 32'h0a5d7728; end
            8'd83: begin cos_pairs_q1_7_o = 32'h7f56c920; sin_pairs_q1_7_o = 32'h0b5e737b; end
            8'd84: begin cos_pairs_q1_7_o = 32'h7f55beaa; sin_pairs_q1_7_o = 32'h0b5f6d5d; end
            8'd85: begin cos_pairs_q1_7_o = 32'h7f54b483; sin_pairs_q1_7_o = 32'h0b5f65ea; end
            8'd86: begin cos_pairs_q1_7_o = 32'h7f53aacf; sin_pairs_q1_7_o = 32'h0b605d8b; end
            8'd87: begin cos_pairs_q1_7_o = 32'h7f52a148; sin_pairs_q1_7_o = 32'h0b615498; end
            8'd88: begin cos_pairs_q1_7_o = 32'h7f51997f; sin_pairs_q1_7_o = 32'h0b624a04; end
            8'd89: begin cos_pairs_q1_7_o = 32'h7e509241; sin_pairs_q1_7_o = 32'h0b63406d; end
            8'd90: begin cos_pairs_q1_7_o = 32'h7e4f8cc7; sin_pairs_q1_7_o = 32'h0b633472; end
            8'd91: begin cos_pairs_q1_7_o = 32'h7e4e8882; sin_pairs_q1_7_o = 32'h0c64290d; end
            8'd92: begin cos_pairs_q1_7_o = 32'h7e4d84b0; sin_pairs_q1_7_o = 32'h0c651c9d; end
            8'd93: begin cos_pairs_q1_7_o = 32'h7e4c8228; sin_pairs_q1_7_o = 32'h0c661088; end
            8'd94: begin cos_pairs_q1_7_o = 32'h7e4b817b; sin_pairs_q1_7_o = 32'h0c6703e1; end
            8'd95: begin cos_pairs_q1_7_o = 32'h7e4a815d; sin_pairs_q1_7_o = 32'h0c67f657; end
            8'd96: begin cos_pairs_q1_7_o = 32'h7e4983e9; sin_pairs_q1_7_o = 32'h0c68ea7d; end
            8'd97: begin cos_pairs_q1_7_o = 32'h7e48868b; sin_pairs_q1_7_o = 32'h0c69dd30; end
            8'd98: begin cos_pairs_q1_7_o = 32'h7e478a98; sin_pairs_q1_7_o = 32'h0c69d1b7; end
            8'd99: begin cos_pairs_q1_7_o = 32'h7e468f05; sin_pairs_q1_7_o = 32'h0d6ac681; end
            8'd100: begin cos_pairs_q1_7_o = 32'h7e45956e; sin_pairs_q1_7_o = 32'h0d6bbbc0; end
            8'd101: begin cos_pairs_q1_7_o = 32'h7e449d71; sin_pairs_q1_7_o = 32'h0d6cb139; end
            8'd102: begin cos_pairs_q1_7_o = 32'h7e42a50d; sin_pairs_q1_7_o = 32'h0d6ca77e; end
            8'd103: begin cos_pairs_q1_7_o = 32'h7e41af9d; sin_pairs_q1_7_o = 32'h0d6d9f4f; end
            8'd104: begin cos_pairs_q1_7_o = 32'h7e40b988; sin_pairs_q1_7_o = 32'h0d6e97d7; end
            8'd105: begin cos_pairs_q1_7_o = 32'h7e3fc4e1; sin_pairs_q1_7_o = 32'h0d6e9085; end
            8'd106: begin cos_pairs_q1_7_o = 32'h7e3ecf57; sin_pairs_q1_7_o = 32'h0d6f8ba4; end
            8'd107: begin cos_pairs_q1_7_o = 32'h7e3ddb7d; sin_pairs_q1_7_o = 32'h0e6f8717; end
            8'd108: begin cos_pairs_q1_7_o = 32'h7e3ce730; sin_pairs_q1_7_o = 32'h0e708376; end
            8'd109: begin cos_pairs_q1_7_o = 32'h7e3bf4b7; sin_pairs_q1_7_o = 32'h0e718268; end
            8'd110: begin cos_pairs_q1_7_o = 32'h7e3a0181; sin_pairs_q1_7_o = 32'h0e7181fa; end
            8'd111: begin cos_pairs_q1_7_o = 32'h7e380dc0; sin_pairs_q1_7_o = 32'h0e728292; end
            8'd112: begin cos_pairs_q1_7_o = 32'h7e371a3a; sin_pairs_q1_7_o = 32'h0e72848f; end
            8'd113: begin cos_pairs_q1_7_o = 32'h7e36267e; sin_pairs_q1_7_o = 32'h0e7387f4; end
            8'd114: begin cos_pairs_q1_7_o = 32'h7e35324f; sin_pairs_q1_7_o = 32'h0e738b64; end
            8'd115: begin cos_pairs_q1_7_o = 32'h7e343dd7; sin_pairs_q1_7_o = 32'h0f749178; end
            8'd116: begin cos_pairs_q1_7_o = 32'h7e334885; sin_pairs_q1_7_o = 32'h0f74981e; end
            8'd117: begin cos_pairs_q1_7_o = 32'h7e3252a4; sin_pairs_q1_7_o = 32'h0f759fa8; end
            8'd118: begin cos_pairs_q1_7_o = 32'h7e305b18; sin_pairs_q1_7_o = 32'h0f75a883; end
            8'd119: begin cos_pairs_q1_7_o = 32'h7e2f6476; sin_pairs_q1_7_o = 32'h0f76b1d1; end
            8'd120: begin cos_pairs_q1_7_o = 32'h7e2e6b67; sin_pairs_q1_7_o = 32'h0f76bc4a; end
            8'd121: begin cos_pairs_q1_7_o = 32'h7e2d71fa; sin_pairs_q1_7_o = 32'h0f77c77f; end
            8'd122: begin cos_pairs_q1_7_o = 32'h7e2c7792; sin_pairs_q1_7_o = 32'h0f77d33f; end
            8'd123: begin cos_pairs_q1_7_o = 32'h7e2a7b8f; sin_pairs_q1_7_o = 32'h1078dfc6; end
            8'd124: begin cos_pairs_q1_7_o = 32'h7e297df4; sin_pairs_q1_7_o = 32'h1078eb82; end
            8'd125: begin cos_pairs_q1_7_o = 32'h7e287f64; sin_pairs_q1_7_o = 32'h1079f8b2; end
            8'd126: begin cos_pairs_q1_7_o = 32'h7e277f78; sin_pairs_q1_7_o = 32'h1079042a; end
            8'd127: begin cos_pairs_q1_7_o = 32'h7e267e1e; sin_pairs_q1_7_o = 32'h1079117c; end
            8'd128: begin cos_pairs_q1_7_o = 32'h7e247ca8; sin_pairs_q1_7_o = 32'h107a1d5c; end
            8'd129: begin cos_pairs_q1_7_o = 32'h7e237883; sin_pairs_q1_7_o = 32'h107a2ae7; end
            8'd130: begin cos_pairs_q1_7_o = 32'h7e2273d1; sin_pairs_q1_7_o = 32'h107a358a; end
            8'd131: begin cos_pairs_q1_7_o = 32'h7e216d4a; sin_pairs_q1_7_o = 32'h117b4199; end
            8'd132: begin cos_pairs_q1_7_o = 32'h7e20667f; sin_pairs_q1_7_o = 32'h117b4b07; end
            8'd133: begin cos_pairs_q1_7_o = 32'h7e1e5e3f; sin_pairs_q1_7_o = 32'h117b556e; end
            8'd134: begin cos_pairs_q1_7_o = 32'h7e1d55c5; sin_pairs_q1_7_o = 32'h117c5e71; end
            8'd135: begin cos_pairs_q1_7_o = 32'h7e1c4c81; sin_pairs_q1_7_o = 32'h117c660b; end
            8'd136: begin cos_pairs_q1_7_o = 32'h7e1b41b2; sin_pairs_q1_7_o = 32'h117c6d9c; end
            8'd137: begin cos_pairs_q1_7_o = 32'h7e19362a; sin_pairs_q1_7_o = 32'h117c7388; end
            8'd138: begin cos_pairs_q1_7_o = 32'h7e182a7c; sin_pairs_q1_7_o = 32'h117d78e3; end
            8'd139: begin cos_pairs_q1_7_o = 32'h7e171e5b; sin_pairs_q1_7_o = 32'h127d7b58; end
            8'd140: begin cos_pairs_q1_7_o = 32'h7e1611e7; sin_pairs_q1_7_o = 32'h127d7e7c; end
            8'd141: begin cos_pairs_q1_7_o = 32'h7e14058a; sin_pairs_q1_7_o = 32'h127d7f2e; end
            8'd142: begin cos_pairs_q1_7_o = 32'h7e13f899; sin_pairs_q1_7_o = 32'h127e7fb5; end
            8'd143: begin cos_pairs_q1_7_o = 32'h7e12eb07; sin_pairs_q1_7_o = 32'h127e7d81; end
            8'd144: begin cos_pairs_q1_7_o = 32'h7e11df6f; sin_pairs_q1_7_o = 32'h127e7bc2; end
            8'd145: begin cos_pairs_q1_7_o = 32'h7e0fd370; sin_pairs_q1_7_o = 32'h127e773b; end
            8'd146: begin cos_pairs_q1_7_o = 32'h7e0ec70b; sin_pairs_q1_7_o = 32'h127e727f; end
            8'd147: begin cos_pairs_q1_7_o = 32'h7e0dbc9b; sin_pairs_q1_7_o = 32'h137e6b4d; end
            8'd148: begin cos_pairs_q1_7_o = 32'h7e0cb288; sin_pairs_q1_7_o = 32'h137e64d5; end
            8'd149: begin cos_pairs_q1_7_o = 32'h7e0aa8e4; sin_pairs_q1_7_o = 32'h137f5c84; end
            8'd150: begin cos_pairs_q1_7_o = 32'h7e09a059; sin_pairs_q1_7_o = 32'h137f53a5; end
            8'd151: begin cos_pairs_q1_7_o = 32'h7e08987c; sin_pairs_q1_7_o = 32'h137f491a; end
            8'd152: begin cos_pairs_q1_7_o = 32'h7e06912e; sin_pairs_q1_7_o = 32'h137f3e77; end
            8'd153: begin cos_pairs_q1_7_o = 32'h7e058bb5; sin_pairs_q1_7_o = 32'h137f3266; end
            8'd154: begin cos_pairs_q1_7_o = 32'h7d048781; sin_pairs_q1_7_o = 32'h137f26f8; end
            8'd155: begin cos_pairs_q1_7_o = 32'h7d0384c2; sin_pairs_q1_7_o = 32'h147f1a91; end
            8'd156: begin cos_pairs_q1_7_o = 32'h7d01823c; sin_pairs_q1_7_o = 32'h147f0e90; end
            8'd157: begin cos_pairs_q1_7_o = 32'h7d00817f; sin_pairs_q1_7_o = 32'h147f01f6; end
            8'd158: begin cos_pairs_q1_7_o = 32'h7dff824d; sin_pairs_q1_7_o = 32'h147ff465; end
            8'd159: begin cos_pairs_q1_7_o = 32'h7dfe83d5; sin_pairs_q1_7_o = 32'h147fe877; end
            8'd160: begin cos_pairs_q1_7_o = 32'h7dfc8684; sin_pairs_q1_7_o = 32'h147fdb1c; end
            8'd161: begin cos_pairs_q1_7_o = 32'h7dfb8ba6; sin_pairs_q1_7_o = 32'h147fcfa7; end
            8'd162: begin cos_pairs_q1_7_o = 32'h7dfa901a; sin_pairs_q1_7_o = 32'h147fc484; end
            8'd163: begin cos_pairs_q1_7_o = 32'h7df89777; sin_pairs_q1_7_o = 32'h157fb9d3; end
            8'd164: begin cos_pairs_q1_7_o = 32'h7df79e66; sin_pairs_q1_7_o = 32'h157faf4c; end
            8'd165: begin cos_pairs_q1_7_o = 32'h7df6a7f8; sin_pairs_q1_7_o = 32'h157fa67f; end
            8'd166: begin cos_pairs_q1_7_o = 32'h7df5b091; sin_pairs_q1_7_o = 32'h157e9d3d; end
            8'd167: begin cos_pairs_q1_7_o = 32'h7df3bb90; sin_pairs_q1_7_o = 32'h157e96c4; end
            8'd168: begin cos_pairs_q1_7_o = 32'h7df2c5f6; sin_pairs_q1_7_o = 32'h157e8f81; end
            8'd169: begin cos_pairs_q1_7_o = 32'h7df1d165; sin_pairs_q1_7_o = 32'h157e8ab4; end
            8'd170: begin cos_pairs_q1_7_o = 32'h7df0dd77; sin_pairs_q1_7_o = 32'h157e862c; end
            8'd171: begin cos_pairs_q1_7_o = 32'h7deee91b; sin_pairs_q1_7_o = 32'h167e837c; end
            8'd172: begin cos_pairs_q1_7_o = 32'h7dedf6a6; sin_pairs_q1_7_o = 32'h167e815a; end
            8'd173: begin cos_pairs_q1_7_o = 32'h7dec0384; sin_pairs_q1_7_o = 32'h167d81e5; end
            8'd174: begin cos_pairs_q1_7_o = 32'h7deb0fd3; sin_pairs_q1_7_o = 32'h167d8289; end
            8'd175: begin cos_pairs_q1_7_o = 32'h7de91c4c; sin_pairs_q1_7_o = 32'h167d849a; end
            8'd176: begin cos_pairs_q1_7_o = 32'h7de8287f; sin_pairs_q1_7_o = 32'h167d8709; end
            8'd177: begin cos_pairs_q1_7_o = 32'h7de7343d; sin_pairs_q1_7_o = 32'h167c8c6f; end
            8'd178: begin cos_pairs_q1_7_o = 32'h7de63fc3; sin_pairs_q1_7_o = 32'h167c926f; end
            8'd179: begin cos_pairs_q1_7_o = 32'h7de44a81; sin_pairs_q1_7_o = 32'h177c9909; end
            8'd180: begin cos_pairs_q1_7_o = 32'h7de354b4; sin_pairs_q1_7_o = 32'h177ca19a; end
            8'd181: begin cos_pairs_q1_7_o = 32'h7de25d2d; sin_pairs_q1_7_o = 32'h177ba989; end
            8'd182: begin cos_pairs_q1_7_o = 32'h7de1657c; sin_pairs_q1_7_o = 32'h177bb3e5; end
            8'd183: begin cos_pairs_q1_7_o = 32'h7ddf6c5a; sin_pairs_q1_7_o = 32'h177bbe5a; end
            8'd184: begin cos_pairs_q1_7_o = 32'h7dde72e5; sin_pairs_q1_7_o = 32'h177ac97c; end
            8'd185: begin cos_pairs_q1_7_o = 32'h7ddd7789; sin_pairs_q1_7_o = 32'h177ad52c; end
            8'd186: begin cos_pairs_q1_7_o = 32'h7ddc7b9b; sin_pairs_q1_7_o = 32'h177ae1b4; end
            8'd187: begin cos_pairs_q1_7_o = 32'h7ddb7e0a; sin_pairs_q1_7_o = 32'h1879ed81; end
            8'd188: begin cos_pairs_q1_7_o = 32'h7dd97f70; sin_pairs_q1_7_o = 32'h1879fac4; end
            8'd189: begin cos_pairs_q1_7_o = 32'h7dd87f6f; sin_pairs_q1_7_o = 32'h1879063d; end
            8'd190: begin cos_pairs_q1_7_o = 32'h7dd77e08; sin_pairs_q1_7_o = 32'h1878137f; end
            8'd191: begin cos_pairs_q1_7_o = 32'h7dd67b9a; sin_pairs_q1_7_o = 32'h18781f4c; end
            8'd192: begin cos_pairs_q1_7_o = 32'h7dd57789; sin_pairs_q1_7_o = 32'h18772cd3; end
            8'd193: begin cos_pairs_q1_7_o = 32'h7dd372e6; sin_pairs_q1_7_o = 32'h18773784; end
            8'd194: begin cos_pairs_q1_7_o = 32'h7dd26c5a; sin_pairs_q1_7_o = 32'h187642a7; end
            8'd195: begin cos_pairs_q1_7_o = 32'h7dd1657c; sin_pairs_q1_7_o = 32'h19764d1c; end
            8'd196: begin cos_pairs_q1_7_o = 32'h7dd05d2b; sin_pairs_q1_7_o = 32'h19765777; end
            8'd197: begin cos_pairs_q1_7_o = 32'h7dcf54b3; sin_pairs_q1_7_o = 32'h19755f65; end
            8'd198: begin cos_pairs_q1_7_o = 32'h7dcd4a81; sin_pairs_q1_7_o = 32'h197567f6; end
            8'd199: begin cos_pairs_q1_7_o = 32'h7ccc3fc4; sin_pairs_q1_7_o = 32'h19746e90; end
            8'd200: begin cos_pairs_q1_7_o = 32'h7ccb343e; sin_pairs_q1_7_o = 32'h19737491; end
            8'd201: begin cos_pairs_q1_7_o = 32'h7cca287f; sin_pairs_q1_7_o = 32'h197379f8; end
            8'd202: begin cos_pairs_q1_7_o = 32'h7cc91c4b; sin_pairs_q1_7_o = 32'h19727c66; end
            8'd203: begin cos_pairs_q1_7_o = 32'h7cc80fd2; sin_pairs_q1_7_o = 32'h1a727e77; end
            8'd204: begin cos_pairs_q1_7_o = 32'h7cc70384; sin_pairs_q1_7_o = 32'h1a717f1a; end
            8'd205: begin cos_pairs_q1_7_o = 32'h7cc5f6a7; sin_pairs_q1_7_o = 32'h1a717fa5; end
            8'd206: begin cos_pairs_q1_7_o = 32'h7cc4e91c; sin_pairs_q1_7_o = 32'h1a707d84; end
            8'd207: begin cos_pairs_q1_7_o = 32'h7cc3dd78; sin_pairs_q1_7_o = 32'h1a707ad5; end
            8'd208: begin cos_pairs_q1_7_o = 32'h7cc2d165; sin_pairs_q1_7_o = 32'h1a6f764d; end
            8'd209: begin cos_pairs_q1_7_o = 32'h7cc1c5f5; sin_pairs_q1_7_o = 32'h1a6e717f; end
            8'd210: begin cos_pairs_q1_7_o = 32'h7cc0ba90; sin_pairs_q1_7_o = 32'h1a6e6a3b; end
            8'd211: begin cos_pairs_q1_7_o = 32'h7cbfb091; sin_pairs_q1_7_o = 32'h1b6d63c2; end
            8'd212: begin cos_pairs_q1_7_o = 32'h7cbea7f9; sin_pairs_q1_7_o = 32'h1b6c5a81; end
            8'd213: begin cos_pairs_q1_7_o = 32'h7cbd9e67; sin_pairs_q1_7_o = 32'h1b6c51b5; end
            8'd214: begin cos_pairs_q1_7_o = 32'h7cbc9776; sin_pairs_q1_7_o = 32'h1b6b472e; end
            8'd215: begin cos_pairs_q1_7_o = 32'h7cba9019; sin_pairs_q1_7_o = 32'h1b6a3c7c; end
            8'd216: begin cos_pairs_q1_7_o = 32'h7cb98ba5; sin_pairs_q1_7_o = 32'h1b6a3058; end
            8'd217: begin cos_pairs_q1_7_o = 32'h7cb88684; sin_pairs_q1_7_o = 32'h1b6924e3; end
            8'd218: begin cos_pairs_q1_7_o = 32'h7cb783d6; sin_pairs_q1_7_o = 32'h1b681888; end
            8'd219: begin cos_pairs_q1_7_o = 32'h7cb6824e; sin_pairs_q1_7_o = 32'h1c670c9c; end
            8'd220: begin cos_pairs_q1_7_o = 32'h7cb5817f; sin_pairs_q1_7_o = 32'h1c67ff0b; end
            8'd221: begin cos_pairs_q1_7_o = 32'h7cb4823b; sin_pairs_q1_7_o = 32'h1c66f271; end
            8'd222: begin cos_pairs_q1_7_o = 32'h7cb384c1; sin_pairs_q1_7_o = 32'h1c65e66e; end
            8'd223: begin cos_pairs_q1_7_o = 32'h7cb28781; sin_pairs_q1_7_o = 32'h1c64d907; end
            8'd224: begin cos_pairs_q1_7_o = 32'h7cb18bb6; sin_pairs_q1_7_o = 32'h1c64ce99; end
            8'd225: begin cos_pairs_q1_7_o = 32'h7cb0912f; sin_pairs_q1_7_o = 32'h1c63c28a; end
            8'd226: begin cos_pairs_q1_7_o = 32'h7caf987d; sin_pairs_q1_7_o = 32'h1c62b7e7; end
            8'd227: begin cos_pairs_q1_7_o = 32'h7caea058; sin_pairs_q1_7_o = 32'h1d61ad5c; end
            8'd228: begin cos_pairs_q1_7_o = 32'h7cada8e2; sin_pairs_q1_7_o = 32'h1d60a47c; end
            8'd229: begin cos_pairs_q1_7_o = 32'h7cacb288; sin_pairs_q1_7_o = 32'h1d609c2a; end
            8'd230: begin cos_pairs_q1_7_o = 32'h7cabbc9c; sin_pairs_q1_7_o = 32'h1d5f95b2; end
            8'd231: begin cos_pairs_q1_7_o = 32'h7caac70c; sin_pairs_q1_7_o = 32'h1d5e8e82; end
            8'd232: begin cos_pairs_q1_7_o = 32'h7caad371; sin_pairs_q1_7_o = 32'h1d5d89c6; end
            8'd233: begin cos_pairs_q1_7_o = 32'h7ca9df6e; sin_pairs_q1_7_o = 32'h1d5c853f; end
            8'd234: begin cos_pairs_q1_7_o = 32'h7ca8ec06; sin_pairs_q1_7_o = 32'h1d5b837f; end
            8'd235: begin cos_pairs_q1_7_o = 32'h7ca7f899; sin_pairs_q1_7_o = 32'h1e5a814a; end
            8'd236: begin cos_pairs_q1_7_o = 32'h7ba6058a; sin_pairs_q1_7_o = 32'h1e5981d1; end
            8'd237: begin cos_pairs_q1_7_o = 32'h7ba511e8; sin_pairs_q1_7_o = 32'h1e598283; end
            8'd238: begin cos_pairs_q1_7_o = 32'h7ba41e5c; sin_pairs_q1_7_o = 32'h1e5885a8; end
            8'd239: begin cos_pairs_q1_7_o = 32'h7ba32a7b; sin_pairs_q1_7_o = 32'h1e57881e; end
            8'd240: begin cos_pairs_q1_7_o = 32'h7ba23629; sin_pairs_q1_7_o = 32'h1e568d78; end
            8'd241: begin cos_pairs_q1_7_o = 32'h7ba141b1; sin_pairs_q1_7_o = 32'h1e559364; end
            8'd242: begin cos_pairs_q1_7_o = 32'h7ba14c82; sin_pairs_q1_7_o = 32'h1e549af4; end
            8'd243: begin cos_pairs_q1_7_o = 32'h7ba055c6; sin_pairs_q1_7_o = 32'h1f53a28f; end
            8'd244: begin cos_pairs_q1_7_o = 32'h7b9f5e40; sin_pairs_q1_7_o = 32'h1f52ab92; end
            8'd245: begin cos_pairs_q1_7_o = 32'h7b9e667f; sin_pairs_q1_7_o = 32'h1f51b5fa; end
            8'd246: begin cos_pairs_q1_7_o = 32'h7b9d6d49; sin_pairs_q1_7_o = 32'h1f50bf68; end
            8'd247: begin cos_pairs_q1_7_o = 32'h7b9d73d0; sin_pairs_q1_7_o = 32'h1f4fcb76; end
            8'd248: begin cos_pairs_q1_7_o = 32'h7b9c7883; sin_pairs_q1_7_o = 32'h1f4ed717; end
            8'd249: begin cos_pairs_q1_7_o = 32'h7b9b7ca9; sin_pairs_q1_7_o = 32'h1f4de3a4; end
            8'd250: begin cos_pairs_q1_7_o = 32'h7b9a7e1f; sin_pairs_q1_7_o = 32'h1f4cef85; end
            8'd251: begin cos_pairs_q1_7_o = 32'h7b997f78; sin_pairs_q1_7_o = 32'h204bfcd7; end
            8'd252: begin cos_pairs_q1_7_o = 32'h7b997f63; sin_pairs_q1_7_o = 32'h204a094f; end
            8'd253: begin cos_pairs_q1_7_o = 32'h7b987df3; sin_pairs_q1_7_o = 32'h2049157e; end
            8'd254: begin cos_pairs_q1_7_o = 32'h7b977a8f; sin_pairs_q1_7_o = 32'h20482239; end
            8'd255: begin cos_pairs_q1_7_o = 32'h7b977792; sin_pairs_q1_7_o = 32'h20472ec0; end
            default: begin cos_pairs_q1_7_o = 32'h7f7f7f7f; sin_pairs_q1_7_o = 32'h00000000; end
        endcase
    end
endmodule


// Fixed-point VXM RoPE (Rotary Position Embedding) stage.
//
// Computes 2D vector rotations on adjacent element pairs using 32-bit signed
// fixed-point inputs (x_in) and 8-bit signed fixed-point trig values (cos_q1_7/sin_q1_7).
// Intermediate 40-bit products are combined and arithmetic right-shifted by 7 (>>> 7).
module vxm_rope #(
    parameter int LANES  = 8,
    parameter int LANE_W = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start_i,
    input  logic [LANES*LANE_W-1:0] x_in,
    input  logic [LANES*8-1:0]      cos_q1_7, // 8-lane signed int8 fixed-point cos (Q1.7)
    input  logic [LANES*8-1:0]      sin_q1_7, // 8-lane signed int8 fixed-point sin (Q1.7)
    output logic [LANES*LANE_W-1:0] y_out,
    output logic                    done_o,
    output logic                    busy_o
);

    logic [LANES*LANE_W-1:0] y_next;
    logic                    busy_q;

    genvar i;
    generate
        for (i = 0; i < LANES/2; i++) begin : g_rope_pairs
            localparam int EVEN = 2 * i;
            localparam int ODD  = 2 * i + 1;

            logic signed [LANE_W-1:0] x_even;
            logic signed [LANE_W-1:0] x_odd;
            logic signed [7:0]        c_val;
            logic signed [7:0]        s_val;

`ifdef LPULITE_VXM_LOGIC_MULT
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod0;
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod1;
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod2;
            (* multstyle = "logic" *) logic signed [LANE_W+8-1:0] prod3;
`else
            logic signed [LANE_W+8-1:0] prod0;
            logic signed [LANE_W+8-1:0] prod1;
            logic signed [LANE_W+8-1:0] prod2;
            logic signed [LANE_W+8-1:0] prod3;
`endif

            logic signed [LANE_W+8-1:0] res_even_full;
            logic signed [LANE_W+8-1:0] res_odd_full;

            logic signed [LANE_W-1:0]   res_even;
            logic signed [LANE_W-1:0]   res_odd;

            assign x_even = $signed(x_in[EVEN*LANE_W +: LANE_W]);
            assign x_odd  = $signed(x_in[ODD*LANE_W  +: LANE_W]);
            assign c_val  = $signed(cos_q1_7[EVEN*8   +: 8]);
            assign s_val  = $signed(sin_q1_7[EVEN*8   +: 8]);

            assign prod0 = x_even * c_val;
            assign prod1 = x_odd  * s_val;
            assign prod2 = x_even * s_val;
            assign prod3 = x_odd  * c_val;

            assign res_even_full = prod0 - prod1;
            assign res_odd_full  = prod2 + prod3;

            // Arithmetic right shift by 7 for Q1.7 fixed-point scaling
            assign res_even = LANE_W'(res_even_full >>> 7);
            assign res_odd  = LANE_W'(res_odd_full  >>> 7);

            assign y_next[EVEN*LANE_W +: LANE_W] = res_even;
            assign y_next[ODD*LANE_W  +: LANE_W] = res_odd;
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out  <= '0;
            done_o <= 1'b0;
            busy_q <= 1'b0;
        end else begin
            done_o <= start_i;
            if (start_i) begin
                y_out  <= y_next;
                busy_q <= 1'b1;
            end else begin
                busy_q <= 1'b0;
            end
        end
    end

    assign busy_o = start_i || busy_q;

endmodule
