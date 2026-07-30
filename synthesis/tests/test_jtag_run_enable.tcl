# One-shot physical-DE1-SoC Avalon-MM smoke test.
# It verifies that the synthesized LPU control register accepts and returns a
# run-enable write; it does not claim to validate transformer inference.
refresh_connections
set masters [get_service_paths master]
if {[llength $masters] == 0} {
    error "ERROR_NO_JTAG_MASTER"
}
set master_path [lindex $masters 0]
open_service master $master_path

master_write_32 $master_path 0xC000 0x1
set asserted [master_read_32 $master_path 0xC000 1]
master_write_32 $master_path 0xC000 0x0
set cleared [master_read_32 $master_path 0xC000 1]

close_service master $master_path
puts "RUN_ENABLE_ASSERTED: $asserted"
puts "RUN_ENABLE_CLEARED: $cleared"
if {$asserted != 1 || $cleared != 0} {
    error "ERROR_RUN_ENABLE_READBACK"
}
