`ifndef CVFPU_COMPAT_SVH
`define CVFPU_COMPAT_SVH

function automatic logic [63:0] f32_to_f64_bits(logic [31:0] f32);
    logic sign;
    logic [7:0] exp32;
    logic [22:0] mant32;
    logic [10:0] exp64;
    logic [51:0] mant64;
    begin
        sign = f32[31];
        exp32 = f32[30:23];
        mant32 = f32[22:0];
        
        if (exp32 == 8'h00) begin
            exp64 = 11'h000;
            mant64 = {mant32, 29'b0};
        end else if (exp32 == 8'hFF) begin
            exp64 = 11'h7FF;
            mant64 = {mant32, 29'b0};
        end else begin
            exp64 = 11'(exp32) - 11'd127 + 11'd1023;
            mant64 = {mant32, 29'b0};
        end
        
        f32_to_f64_bits = {sign, exp64, mant64};
    end
endfunction

function automatic logic [31:0] f64_to_f32_bits(logic [63:0] f64);
    logic sign;
    logic [10:0] exp64;
    logic [51:0] mant64;
    logic [7:0] exp32;
    logic [22:0] mant32;
    int biased_exp;
    begin
        sign = f64[63];
        exp64 = f64[62:52];
        mant64 = f64[51:0];
        
        if (exp64 == 11'h000) begin
            exp32 = 8'h00;
            mant32 = mant64[51:29];
        end else if (exp64 == 11'h7FF) begin
            exp32 = 8'hFF;
            mant32 = mant64[51:29];
        end else begin
            biased_exp = int'(exp64) - 1023 + 127;
            if (biased_exp <= 0) begin
                exp32 = 8'h00;
                mant32 = 23'h0;
            end else if (biased_exp >= 255) begin
                exp32 = 8'hFF;
                mant32 = 23'h0;
            end else begin
                exp32 = 8'(biased_exp);
                mant32 = mant64[51:29];
            end
        end
        
        f64_to_f32_bits = {sign, exp32, mant32};
    end
endfunction

`endif
