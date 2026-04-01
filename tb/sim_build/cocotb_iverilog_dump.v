module cocotb_iverilog_dump();
initial begin
    $dumpfile("sim_build/vxm.fst");
    $dumpvars(0, vxm);
end
endmodule
