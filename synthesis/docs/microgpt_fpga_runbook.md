# LPULite MicroGPT FPGA runbook

This is the canonical setup procedure for the working ARM+FPGA MicroGPT
configuration on the DE1-SoC.

## Verified release files

- FPGA bitstream:
  `synthesis/build/lpu_lite_de1_soc_hps/lpu_lite_de1_soc_hps.sof`
- Expected SHA-256:
  `8102D2F0DFEE19AC8F49AC1335D4AB93A9914107E071829572B7188DE692D02B`
- ARM runtime:
  `synthesis/linux/microgpt_hps_runtime`
- Quartus project:
  `synthesis/build/lpu_lite_de1_soc_hps/lpu_lite_de1_soc_hps.qpf`

The `.sof` is volatile. Program it again after every board power cycle unless
FPGA configuration is later added to the Linux boot process. The runtime loads
the model weights into FPGA memory whenever it starts.

## 1. Required hardware

- Powered DE1-SoC with the Linux SD card inserted.
- USB-Blaster/JTAG cable for loading the `.sof`.
- USB UART cable for the Linux serial console.
- Ethernet cable directly between the laptop and DE1-SoC.

The USB UART and Ethernet connections do not configure the FPGA. JTAG loads
the FPGA hardware, UART provides the recovery/login console, and Ethernet is
used for SSH/SCP.

## 2. Required laptop software

- Quartus Prime 25.1 Standard Lite at
  `C:\altera_lite\25.1std\quartus\bin64`.
- Windows OpenSSH client (`ssh` and `scp`).
- WSL Ubuntu with the ARM hard-float cross compiler when rebuilding the ARM
  runtime.
- PuTTY or another serial terminal.

Install the WSL compiler once from the WSL shell:

```sh
sudo apt update
sudo apt install -y make gcc-arm-linux-gnueabihf
```

## 3. Optional: rebuild everything from source

The repository contains this release build script:

```text
synthesis/scripts/build_microgpt_fpga_release.ps1
```

Run it from Windows PowerShell at the repository root:

```powershell
cd "C:\Users\micha\Documents(Local)\Projects\LPULite"
Set-ExecutionPolicy -Scope Process Bypass
.\synthesis\scripts\build_microgpt_fpga_release.ps1
```

The script performs the following operations:

1. Compiles the INT8 checkpoint into MEM1 and VLIW images.
2. Exports those images to `microgpt_hps_image.h`.
3. Cross-compiles the static ARM runtime in WSL.
4. Maps the repository to `T:` to avoid Quartus path problems.
5. Compiles the existing HPS Quartus project.
6. Prints the resulting `.sof` SHA-256.

Useful software-only variants:

```powershell
# Rebuild model artifacts and the ARM runtime without spending 20 minutes in Quartus.
.\synthesis\scripts\build_microgpt_fpga_release.ps1 -SkipQuartus

# Recompile only the Quartus project using the already generated model/runtime files.
.\synthesis\scripts\build_microgpt_fpga_release.ps1 -SkipModelArtifacts -SkipArmRuntime
```

Manual equivalent commands are:

```powershell
cd "C:\Users\micha\Documents(Local)\Projects\LPULite"

python model\tools\compile_microgpt_lpu.py
python synthesis\scripts\export_microgpt_hps_headers.py

subst T: "C:\Users\micha\Documents(Local)\Projects\LPULite"
& "C:\altera_lite\25.1std\quartus\bin64\quartus_sh.exe" `
  --flow compile `
  "T:\synthesis\build\lpu_lite_de1_soc_hps\lpu_lite_de1_soc_hps"
```

Cross-compile the runtime from WSL:

```sh
cd "/mnt/c/Users/micha/Documents(Local)/Projects/LPULite/synthesis/linux"
make clean
make CC=arm-linux-gnueabihf-gcc LDFLAGS=-static
```

Rebuilding is optional for ordinary board setup. The verified `.sof` and ARM
binary are committed to the repository.

## 4. Power and boot the board

1. Insert the Linux SD card.
2. Connect power, USB-Blaster, USB UART and Ethernet.
3. Turn on the board.
4. Wait for Linux to boot.

Find the UART ports from Windows PowerShell:

```powershell
Get-PnpDevice -Class Ports
```

For the Silicon Labs dual CP210x cable used during development, the enhanced
port was `COM4`. The COM number may change on another computer.

Open PuTTY with:

- Connection type: Serial.
- Serial line: the enhanced COM port, for example `COM4`.
- Speed: `115200`.
- Data bits: 8.
- Stop bits: 1.
- Parity: none.
- Flow control: none.

Log in as `root`. The Terasic image may allow an empty password on UART. Set a
root password once if SSH asks for one:

```sh
passwd
```

## 5. Configure the direct Ethernet connection

Open an Administrator Windows PowerShell. Confirm the adapter name:

```powershell
Get-NetAdapter
```

The development adapter was named `Ethernet 3`. Assign the laptop
`192.168.1.10/24`:

```powershell
netsh interface ipv4 set address name="Ethernet 3" static 192.168.1.10 255.255.255.0
ipconfig
```

From the board's UART terminal, configure the board address. This exact command
must normally be repeated after each boot because it is not persistent:

```sh
ifconfig eth0 192.168.1.101 netmask 255.255.255.0 up
ifconfig eth0
```

Test the connection from Windows:

```powershell
ping -S 192.168.1.10 192.168.1.101
ssh root@192.168.1.101
```

If ping reports `Destination host unreachable`, verify that the Ethernet link
LEDs are active and that both interfaces have the addresses above.

## 6. Program the verified FPGA bitstream

The board must be powered on and the USB-Blaster must be connected. From the
repository root in Windows PowerShell:

```powershell
cd "C:\Users\micha\Documents(Local)\Projects\LPULite"

& "C:\altera_lite\25.1std\quartus\bin64\jtagconfig.exe"

& "C:\altera_lite\25.1std\quartus\bin64\quartus_pgm.exe" `
  -m JTAG `
  -c "DE-SoC [USB-1]" `
  -o "p;synthesis\build\lpu_lite_de1_soc_hps\lpu_lite_de1_soc_hps.sof@2"
```

The `@2` is required because the DE1-SoC JTAG chain exposes the ARM debug
device before the Cyclone V FPGA. Without it, Quartus may report FPGA ID
`0x02D120DD` expected but ARM debug ID `0x4BA00477` found.

Successful programming ends with `Quartus Prime Programmer was successful`.

## 7. Copy the ARM runtime to the board

Create the destination directory from UART or SSH:

```sh
mkdir -p /home/root/linux
```

Copy the runtime from Windows PowerShell:

```powershell
cd "C:\Users\micha\Documents(Local)\Projects\LPULite"
scp synthesis\linux\microgpt_hps_runtime root@192.168.1.101:/home/root/linux/
```

Log in and make it executable:

```powershell
ssh root@192.168.1.101
```

```sh
cd /home/root/linux
chmod +x microgpt_hps_runtime
```

## 8. Verify the HPS-to-FPGA bridge

Run the non-destructive bridge probe before inference:

```sh
cd /home/root/linux
./microgpt_hps_runtime --probe-only
```

The expected result includes successful control-register, exact-cycle, MEM0
and MEM1 tests. Stop if the probe fails; the usual causes are an unprogrammed
FPGA, the wrong `.sof`, or an incorrect physical bridge base.

## 9. Run interactive inference

Use the verified production path:

```sh
cd /home/root/linux
./microgpt_hps_runtime \
  --attention fpga-mxm \
  --broadcast host \
  --decode greedy \
  --benchmark
```

Example:

```text
Prompt > sat
satvik
```

In this mode:

- FPGA SXM transposes K.
- FPGA MXM computes QK and PV.
- FPGA VXM computes attention softmax.
- FPGA MXM/VXM execute the remaining transformer arithmetic.
- ARM software manages MMIO, layout, causal length, cache addresses, argmax
  and terminal I/O.

The runtime loads the 974 model rows at startup. Do not use `--no-load-weights`
after a power cycle or FPGA reconfiguration unless the weights have already
been loaded by another runtime invocation.

## 10. Run a repeatable benchmark

Avoid typing delays by using a fixed prompt and repeat count:

```sh
./microgpt_hps_runtime \
  --attention fpga-mxm \
  --broadcast host \
  --decode greedy \
  --benchmark \
  --prompt sat \
  --repeat 10
```

The latest verified result was approximately:

- 41.3 complete LPU forward-pass steps/s.
- 20.7 end-to-end output characters/s for `sat` to `satvik`.
- 145 ms mean prompt-to-completion time.

Use the aggregate line for comparisons. Output tokens/s depends on prompt and
completion length; LPU steps/s is the more stable forward-pass measurement.

## 11. Minimal procedure after every power cycle

1. Boot Linux from the SD card.
2. Open UART and log in as `root`.
3. Run:

   ```sh
   ifconfig eth0 192.168.1.101 netmask 255.255.255.0 up
   ```

4. Program the committed HPS `.sof` over JTAG using the `@2` command above.
5. SSH to `root@192.168.1.101`.
6. Run:

   ```sh
   cd /home/root/linux
   ./microgpt_hps_runtime --attention fpga-mxm --broadcast host --decode greedy --benchmark
   ```

The ARM runtime only needs to be copied again when its executable changes.
