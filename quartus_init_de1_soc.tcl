set root [file normalize [file dirname [info script]]]
set project_dir [file join $root build tiny_lpu_de1_soc]
file mkdir $project_dir
cd $project_dir
project_new -overwrite tiny_lpu_de1_soc
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6N
set_global_assignment -name TOP_LEVEL_ENTITY de1_soc_top
set_global_assignment -name SYSTEMVERILOG_FILE [file join $root src de1_soc_top.sv]
foreach f [glob -nocomplain [file join $root src *.sv]] {
    if {$f ne [file join $root src de1_soc_top.sv] && $f ne [file join $root src lpu_de1_soc_wrapper.sv]} { set_global_assignment -name SYSTEMVERILOG_FILE $f }
}
set_global_assignment -name SEARCH_PATH [file join $root src]
set_location_assignment PIN_AF14 -to CLOCK_50
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50
set_location_assignment PIN_AJ4 -to KEY[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]
set_location_assignment PIN_V16 -to LEDR[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]
project_close
