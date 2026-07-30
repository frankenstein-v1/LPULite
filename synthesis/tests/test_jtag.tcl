set masters [get_service_paths master]
if {[llength $masters] == 0} {
    puts "ERROR_NO_JTAG_MASTER"
    exit 1
}
set master_path [lindex $masters 0]
open_service master $master_path
puts "JTAG_MASTER_OPENED: $master_path"
# Read control register at 0xC000 (0x3000 in word address / 0xC000 in byte address)
set val [master_read_32 $master_path 0xC000 1]
puts "CTRL_REG_READ: $val"
close_service master $master_path
