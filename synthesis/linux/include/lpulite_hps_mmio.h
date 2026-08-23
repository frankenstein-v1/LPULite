#ifndef LPULITE_HPS_MMIO_H
#define LPULITE_HPS_MMIO_H

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
#define LPULITE_HPS_LW_BRIDGE_BASE_DEFAULT 0xFF200000u
#define LPULITE_HPS_LW_BRIDGE_SPAN_DEFAULT 0x00010000u

/* LPU Avalon offsets inside the exported lightweight bridge window. */
#define LPULITE_IMEM_OFFSET    0x0000u
#define LPULITE_MEM0_OFFSET    0x4000u
#define LPULITE_MEM1_OFFSET    0x8000u
#define LPULITE_CTRL_RUN       0xC000u
#define LPULITE_CTRL_PC_LOAD   0xC004u
#define LPULITE_CTRL_CYCLES    0xC008u
#define LPULITE_CTRL_RUN_CYCLES 0xC00Cu
#define LPULITE_CTRL_SOFT_RESET 0xC010u

#define LPULITE_IMEM_ROWS      1024u
#define LPULITE_ROW_BYTES      12u

typedef struct {
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
} lpulite_mmio_row_t;

typedef struct {
    uint64_t read32;
    uint64_t write32;
    uint64_t row_reads;
    uint64_t row_writes;
    uint64_t imem_row_writes;
} lpulite_mmio_stats_t;

typedef struct {
    int fd;
    size_t span;
    uintptr_t phys_base;
    volatile uint8_t *base;
    lpulite_mmio_stats_t stats;
} lpulite_mmio_t;

int lpulite_mmio_open(lpulite_mmio_t *dev, uintptr_t phys_base, size_t span);
void lpulite_mmio_close(lpulite_mmio_t *dev);
lpulite_mmio_stats_t lpulite_mmio_get_stats(const lpulite_mmio_t *dev);

void lpulite_write32(lpulite_mmio_t *dev, uint32_t offset, uint32_t value);
uint32_t lpulite_read32(lpulite_mmio_t *dev, uint32_t offset);
void lpulite_soft_reset(lpulite_mmio_t *dev, uint32_t cycles, unsigned poll_us);

void lpulite_write_row(lpulite_mmio_t *dev, uint32_t base_offset, uint32_t row, lpulite_mmio_row_t value);
lpulite_mmio_row_t lpulite_read_row(lpulite_mmio_t *dev, uint32_t base_offset, uint32_t row);
void lpulite_write_rows(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    const lpulite_mmio_row_t *values,
    size_t count
);
void lpulite_read_rows(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    lpulite_mmio_row_t *values,
    size_t count
);
void lpulite_fill_rows(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    lpulite_mmio_row_t value,
    size_t count
);
void lpulite_copy_row(lpulite_mmio_t *dev, uint32_t src_base, uint32_t src_row, uint32_t dst_base, uint32_t dst_row);

void lpulite_load_imem_page(lpulite_mmio_t *dev, const lpulite_mmio_row_t *program, size_t count);
void lpulite_run_page(lpulite_mmio_t *dev, unsigned settle_us);
uint32_t lpulite_run_cycles(lpulite_mmio_t *dev, uint32_t cycles, unsigned poll_us);
uint32_t lpulite_run_cycles_from(
    lpulite_mmio_t *dev,
    uint32_t pc,
    uint32_t cycles,
    unsigned poll_us
);

#ifdef __cplusplus
}
#endif

#endif /* LPULITE_HPS_MMIO_H */
