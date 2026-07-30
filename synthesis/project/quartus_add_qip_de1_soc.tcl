set project_root [file normalize [file dirname [info script]]]
set synthesis_root [file dirname $project_root]
set project_dir [file join $synthesis_root build tiny_lpu_de1_soc]
cd $project_dir
project_open tiny_lpu_de1_soc
set_global_assignment -name QIP_FILE [file join $project_root platform_designer_system synthesis platform_designer_system.qip]
project_close
