set_module_property NAME lpu_de1_soc
set_module_property VERSION 1.0
set_module_property GROUP "TinyLPU"
set_module_property DISPLAY_NAME "TinyLPU DE1-SoC Avalon wrapper"
set_module_property TOP_LEVEL_HDL_MODULE lpu_de1_soc_wrapper
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
add_interface clk clock end
add_interface_port clk clk clk Input 1
add_interface rst_n reset end
set_interface_property rst_n associatedClock clk
set_interface_property rst_n synchronousEdges DEASSERT
add_interface_port rst_n rst_n reset_n Input 1
add_interface avs avalon end
set_interface_property avs addressUnits SYMBOLS
set_interface_property avs associatedClock clk
set_interface_property avs associatedReset rst_n
set_interface_property avs readWaitTime 0
set_interface_property avs writeWaitTime 0
set_interface_property avs maximumPendingReadTransactions 0
add_interface_port avs avs_address address Input 16
add_interface_port avs avs_read read Input 1
add_interface_port avs avs_write write Input 1
add_interface_port avs avs_writedata writedata Input 32
add_interface_port avs avs_readdata readdata Output 32
add_interface_port avs avs_waitrequest waitrequest Output 1
