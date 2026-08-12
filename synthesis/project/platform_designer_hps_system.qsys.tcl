package require -exact qsys 25.1
create_system platform_designer_hps_system
set_project_property DEVICE_FAMILY "Cyclone V"
set_project_property DEVICE 5CSEMA5F31C6

add_instance clk_0 clock_source
set_instance_parameter_value clk_0 clockFrequency {50000000.0}

add_instance hps_0 altera_hps
set_instance_parameter_value hps_0 S2F_Width 0
set_instance_parameter_value hps_0 F2S_Width 0
set_instance_parameter_value hps_0 LWH2F_Enable true
set_instance_parameter_value hps_0 F2SDRAM_Width {}
set_instance_parameter_value hps_0 F2SDRAM_Type {}
set_instance_parameter_value hps_0 MPU_EVENTS_Enable false
# This is intended to avoid regenerating the SDRAM sequencer for an FPGA-only
# bridge image.  Quartus 25.1 may still invoke the HPS SDRAM generator; if it
# does, install a WSL distro or use a board reference HPS handoff project.
set_instance_parameter_value hps_0 quartus_ini_hps_ip_suppress_sdram_synth true

add_instance lpu_0 lpu_de1_soc

add_connection clk_0.clk hps_0.h2f_lw_axi_clock
add_connection clk_0.clk lpu_0.clk
add_connection clk_0.clk_reset lpu_0.rst_n
add_connection hps_0.h2f_lw_axi_master lpu_0.avs
set_connection_parameter_value hps_0.h2f_lw_axi_master/lpu_0.avs baseAddress 0x00000000

add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_0.clk_in_reset

save_system platform_designer_hps_system.qsys
