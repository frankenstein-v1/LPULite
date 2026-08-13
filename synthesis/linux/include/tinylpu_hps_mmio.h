#ifndef TINYLPU_HPS_MMIO_H
#define TINYLPU_HPS_MMIO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DE1-SoC lightweight HPS-to-FPGA bridge.
 *
 * 0xFF200000 is the common Linux physical base for the lightweight bridge on
 * Cyclone V SoC designs. If your Platform Designer address map differs, pass a
 * different base address to the runtime with --base.
 */
#define TINYLPU_HPS_LW_BRIDGE_BASE_DEFAULT 0xFF200000u
#define TINYLPU_HPS_LW_BRIDGE_SPAN_DEFAULT 0x00010000u

/* LPU Avalon offsets inside the exported lightweight bridge window. */
#define TINYLPU_IMEM_OFFSET    0x0000u
#define TINYLPU_MEM0_OFFSET    0x4000u
#define TINYLPU_MEM1_OFFSET    0x8000u
#define TINYLPU_CTRL_RUN       0xC000u
#define TINYLPU_CTRL_PC_LOAD   0xC004u
#define TINYLPU_CTRL_CYCLES    0xC008u
#define TINYLPU_CTRL_RUN_CYCLES 0xC00Cu
#define TINYLPU_CTRL_SOFT_RESET 0xC010u

#define TINYLPU_IMEM_ROWS      1024u
#define TINYLPU_ROW_BYTES      12u

typedef struct {
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
} tinylpu_mmio_row_t;

typedef struct {
    int fd;
    size_t span;
    uintptr_t phys_base;
    volatile uint8_t *base;
} tinylpu_mmio_t;

int tinylpu_mmio_open(tinylpu_mmio_t *dev, uintptr_t phys_base, size_t span);
void tinylpu_mmio_close(tinylpu_mmio_t *dev);

void tinylpu_write32(tinylpu_mmio_t *dev, uint32_t offset, uint32_t value);
uint32_t tinylpu_read32(tinylpu_mmio_t *dev, uint32_t offset);
void tinylpu_soft_reset(tinylpu_mmio_t *dev, uint32_t cycles, unsigned poll_us);

void tinylpu_write_row(tinylpu_mmio_t *dev, uint32_t base_offset, uint32_t row, tinylpu_mmio_row_t value);
tinylpu_mmio_row_t tinylpu_read_row(tinylpu_mmio_t *dev, uint32_t base_offset, uint32_t row);
void tinylpu_copy_row(tinylpu_mmio_t *dev, uint32_t src_base, uint32_t src_row, uint32_t dst_base, uint32_t dst_row);

void tinylpu_load_imem_page(tinylpu_mmio_t *dev, const tinylpu_mmio_row_t *program, size_t count);
void tinylpu_run_page(tinylpu_mmio_t *dev, unsigned settle_us);
uint32_t tinylpu_run_cycles(tinylpu_mmio_t *dev, uint32_t cycles, unsigned poll_us);
uint32_t tinylpu_run_cycles_from(
    tinylpu_mmio_t *dev,
    uint32_t pc,
    uint32_t cycles,
    unsigned poll_us
);

#ifdef __cplusplus
}
#endif

#endif /* TINYLPU_HPS_MMIO_H */
