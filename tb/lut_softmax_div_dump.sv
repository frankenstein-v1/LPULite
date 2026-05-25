module cocotb_iverilog_dump();
initial begin
    $dumpfile("sim_build/lut_softmax_div.fst");
    $dumpvars(0, lut_softmax_div);
end
endmodule
