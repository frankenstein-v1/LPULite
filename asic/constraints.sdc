create_clock -name core_clk -period 10.000 [get_ports clk]

# Model a nearby host/controller. These constraints are intentionally modest:
# this is a core-level showcase, not a pad-ring or package implementation.
set_input_delay 1.0 -clock core_clk [get_ports {
    rst_n run_en pc_load_en pc_load_value
    ext_en ext_write ext_target ext_addr ext_wdata
}]
set_output_delay 1.0 -clock core_clk [all_outputs]
set_load 0.05 [all_outputs]

set_false_path -from [get_ports rst_n]
