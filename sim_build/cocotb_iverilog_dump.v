module cocotb_iverilog_dump();
initial begin
    string dumpfile_path;    if ($value$plusargs("dumpfile_path=%s", dumpfile_path)) begin
        $dumpfile(dumpfile_path);
    end else begin
        $dumpfile("C:\\Users\\z005a2rv\\Documents\\tinyLPU\\sim_build\\sxm.fst");
    end
    $dumpvars(0, sxm);
end
endmodule
