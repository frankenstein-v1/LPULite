# Prompt to give to Codex on your Quartus Laptop

Copy and paste the prompt below into the AI assistant (Codex/ChatGPT/Gemini) running on the laptop with Quartus Prime Lite.

***

### Paste this into the local AI:

"I am setting up a hardware inference accelerator project for an **Intel DE1-SoC** (Cyclone V, device ID **`5CSEMA5F31C6N`**) using Intel Quartus Prime Lite. 

I have a directory of SystemVerilog design files inside `src/`:
1. `lpu_pkg.sv`: Architectural packages & parameters.
2. `lpu.sv`: Core LPU accelerator.
3. `lpu_de1_soc_wrapper.sv`: Avalon Memory-Mapped bus wrapper.
4. `mem.sv`: SRAM arrays with external read/write ports.
5. Other supporting SV design files.

I want to compile the project entirely using Quartus command-line tools to save time. Please write a PowerShell or Bash automation script that does the following:

1. **Creates a Quartus Project** named `tiny_lpu_de1_soc` targeting the `5CSEMA5F31C6N` chip.
2. **Adds all `.sv` and `.v` files** in the `src/` directory to the project.
3. **Sets `de1_soc_top.sv`** as the top-level entity (please write this simple wrapper file mapping `CLOCK_50` (PIN_AF14), `KEY[0]` (PIN_AJ4), and `LEDR[0]` (PIN_V16)).
4. **Performs Pin Assignments** for the clock, reset key, and LED status pins.
5. **Creates and packages a Platform Designer (Qsys) design** containing:
   * A 50MHz Clock Source.
   * A **JTAG to Avalon Master Bridge** (IP name: `altera_jtag_avalon_master`).
   * Our custom **`lpu_de1_soc_wrapper.sv`** component mapped to base address `0x0000_0000`.
6. **Generates the Qsys system** HDL files.
7. **Runs the full Quartus Compilation Flow** (`quartus_sh --flow compile`) to generate the programming `.sof` bitstream.

Find the Quartus bin directory (defaulting on Windows to `C:\intelFPGA_lite\18.1\quartus\bin64\`) to execute the commands. Outline the steps clearly."

***

## Checklist: What you need on the Quartus Laptop

Before running the automation prompt, ensure you have:

1. **The Codebase Files:** 
   Copy the `src/` folder from this laptop to the Quartus laptop.
2. **The Programming Connection (Optional for compilation):**
   * You **do not** need the board plugged in to design, compile, or generate the `.sof` bitstream.
   * You **only** need the board plugged in during the final step when flashing the design.
3. **Hardware Setup (for Flashing only):**
   * Connect the power cable to the DE1-SoC.
   * Connect the USB-Blaster II USB port on the board to the Quartus laptop.
   * Set the **MSEL[4:0]** switches on the underside/back of the board to **`10101`** (JTAG configuration mode).
