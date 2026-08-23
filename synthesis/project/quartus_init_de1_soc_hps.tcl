set project_root [file normalize [file dirname [info script]]]
set synthesis_root [file dirname $project_root]
set root [file dirname $synthesis_root]
set project_dir [file join $synthesis_root build lpu_lite_de1_soc_hps]
file mkdir $project_dir
cd $project_dir
file delete -force lpu_lite_de1_soc_hps.qsf
project_new -overwrite lpu_lite_de1_soc_hps
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
set_global_assignment -name TOP_LEVEL_ENTITY de1_soc_hps_top
set_global_assignment -name VERILOG_MACRO SYNTHESIS
set_global_assignment -name SYSTEMVERILOG_FILE [file join $synthesis_root rtl de1_soc_hps_top.sv]
set source_dirs [concat [list [file join $root src]] [glob -nocomplain -types d [file join $root src *]]]
foreach dir $source_dirs {
    if {[file tail $dir] ne "archive"} {
        foreach f [glob -nocomplain [file join $dir *.sv]] {
            if {$f ne [file join $root src lpu_pkg.sv]} {
                set_global_assignment -name SYSTEMVERILOG_FILE $f
            }
        }
    }
}
set_global_assignment -name SEARCH_PATH [file join $root src]
set_global_assignment -name SEARCH_PATH [file join $project_root ip lpu_de1_soc]
set_location_assignment PIN_AF14 -to CLOCK_50
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50
set_location_assignment PIN_AJ4 -to KEY[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]
set_location_assignment PIN_V16 -to LEDR[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]
project_close
