/*
 * LPULite Native C Hardware Driver Header
 * DE1-SoC ARM Cortex-A9 Memory-Mapped I/O (MMIO) Interface
 */

#ifndef LPU_DRIVER_H
#define LPU_DRIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Base Address of Cyclone V Lightweight HPS-to-FPGA AXI Bridge
#define LWHPS2FPGA_BASE      0xFF200000UL
#define LWHPS2FPGA_SPAN      0x00010000UL // 64 KB Span

// Hardware Offset Addresses
#define LPU_IMEM_OFFSET      0x0000UL     // 96-bit VLIW Instruction Memory
#define LPU_MEM0_OFFSET      0x4000UL     // Activation Input / Output SRAM
#define LPU_MEM1_OFFSET      0x8000UL     // Quantized Model Weight SRAM
#define LPU_CTRL_OFFSET      0xC000UL     // Control Register (run_enable)

typedef struct {
    volatile uint32_t *imem;
    volatile uint32_t *mem0;
    volatile uint32_t *mem1;
    volatile uint32_t *ctrl;
    bool is_mapped;
} lpu_hardware_t;

// Public Driver Functions
int lpu_init(lpu_hardware_t *lpu);
void lpu_cleanup(lpu_hardware_t *lpu);
void lpu_load_program_and_weights(lpu_hardware_t *lpu);
uint32_t lpu_hardware_step(lpu_hardware_t *lpu, uint32_t last_token_id);
void lpu_generate_story(lpu_hardware_t *lpu, const char *prompt_text, uint32_t max_tokens);

#ifdef __cplusplus
}
#endif

#endif /* LPU_DRIVER_H */
