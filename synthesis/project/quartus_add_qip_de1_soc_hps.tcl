set project_root [file normalize [file dirname [info script]]]
set synthesis_root [file dirname $project_root]
set project_dir [file join $synthesis_root build tiny_lpu_de1_soc_hps]
cd $project_dir
project_open tiny_lpu_de1_soc_hps
set_global_assignment -name QIP_FILE [file join $project_root platform_designer_hps_system synthesis platform_designer_hps_system.qip]
project_close
