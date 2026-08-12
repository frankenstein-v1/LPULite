create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]

set qsys_clk_pins [get_pins -nowarn {*qsys_clk_div[2]*|q}]
if {[llength $qsys_clk_pins] > 0} {
    create_generated_clock -name QSYS_CLK -source [get_ports {CLOCK_50}] -divide_by 8 $qsys_clk_pins
}
