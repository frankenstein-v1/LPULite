set root [file normalize [file dirname [info script]]]
set project_dir [file join $root build tiny_lpu_de1_soc]
cd $project_dir
project_open tiny_lpu_de1_soc
set_global_assignment -name QIP_FILE [file join $root platform_designer_system synthesis platform_designer_system.qip]
project_close
