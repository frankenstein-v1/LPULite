#include "lpulite_hps_mmio.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static volatile uint32_t *word_ptr(lpulite_mmio_t *dev, uint32_t offset) {
    return (volatile uint32_t *)(void *)(dev->base + offset);
}

static void io_barrier(void) {
    __sync_synchronize();
}

static void write_row_unbarriered(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t row,
    lpulite_mmio_row_t value
) {
    uint32_t addr = base_offset + row * LPULITE_ROW_BYTES;
    *word_ptr(dev, addr + 0u) = value.w0;
    *word_ptr(dev, addr + 4u) = value.w1;
    /* Word 2 commits the wrapper's assembled row and must remain last. */
    *word_ptr(dev, addr + 8u) = value.w2;
    dev->stats.write32 += 3u;
    dev->stats.row_writes += 1u;
    if (base_offset == LPULITE_IMEM_OFFSET) {
        dev->stats.imem_row_writes += 1u;
    }
}

static lpulite_mmio_row_t read_row_unbarriered(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t row
) {
    uint32_t addr = base_offset + row * LPULITE_ROW_BYTES;
    /* Prime the wrapper's registered SRAM read path. */
    (void)*word_ptr(dev, addr);
    lpulite_mmio_row_t value = {
        *word_ptr(dev, addr + 0u),
        *word_ptr(dev, addr + 4u),
        *word_ptr(dev, addr + 8u),
    };
    dev->stats.read32 += 4u;
    dev->stats.row_reads += 1u;
    return value;
}

int lpulite_mmio_open(lpulite_mmio_t *dev, uintptr_t phys_base, size_t span) {
    if (!dev) {
        errno = EINVAL;
        return -1;
    }

    dev->fd = -1;
    dev->span = span;
    dev->phys_base = phys_base;
    dev->base = NULL;
    memset(&dev->stats, 0, sizeof(dev->stats));

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
        fprintf(stderr, "Run on the DE1-SoC ARM Linux side, usually with sudo/root.\n");
        return -1;
    }

    void *mapped = mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)phys_base);
    if (mapped == MAP_FAILED) {
        int saved = errno;
        close(fd);
        errno = saved;
        fprintf(stderr, "mmap 0x%08lx failed: %s\n", (unsigned long)phys_base, strerror(errno));
        return -1;
    }

    dev->fd = fd;
    dev->base = (volatile uint8_t *)mapped;
    return 0;
}

lpulite_mmio_stats_t lpulite_mmio_get_stats(const lpulite_mmio_t *dev) {
    return dev->stats;
}

void lpulite_mmio_close(lpulite_mmio_t *dev) {
    if (!dev) {
        return;
    }
    if (dev->base) {
        munmap((void *)dev->base, dev->span);
    }
    if (dev->fd >= 0) {
        close(dev->fd);
    }
    dev->fd = -1;
    dev->base = NULL;
    dev->span = 0;
    dev->phys_base = 0;
}

void lpulite_write32(lpulite_mmio_t *dev, uint32_t offset, uint32_t value) {
    *word_ptr(dev, offset) = value;
    dev->stats.write32 += 1u;
    io_barrier();
}

uint32_t lpulite_read32(lpulite_mmio_t *dev, uint32_t offset) {
    io_barrier();
    uint32_t value = *word_ptr(dev, offset);
    dev->stats.read32 += 1u;
    io_barrier();
    return value;
}

void lpulite_soft_reset(lpulite_mmio_t *dev, uint32_t cycles, unsigned poll_us) {
    if (cycles == 0u) {
        cycles = 16u;
    }
    lpulite_write32(dev, LPULITE_CTRL_RUN, 0u);
    lpulite_write32(dev, LPULITE_CTRL_RUN_CYCLES, 0u);
    lpulite_write32(dev, LPULITE_CTRL_PC_LOAD, 0u);
    lpulite_write32(dev, LPULITE_CTRL_SOFT_RESET, cycles);
    for (unsigned guard = 0; guard < 100000u; ++guard) {
        uint32_t remaining = lpulite_read32(dev, LPULITE_CTRL_SOFT_RESET);
        if (remaining == 0u) {
            break;
        }
        if (poll_us) {
            usleep(poll_us);
        }
    }
}

void lpulite_write_row(lpulite_mmio_t *dev, uint32_t base_offset, uint32_t row, lpulite_mmio_row_t value) {
    write_row_unbarriered(dev, base_offset, row, value);
    io_barrier();
}

lpulite_mmio_row_t lpulite_read_row(lpulite_mmio_t *dev, uint32_t base_offset, uint32_t row) {
    io_barrier();
    lpulite_mmio_row_t value = read_row_unbarriered(dev, base_offset, row);
    io_barrier();
    return value;
}

void lpulite_write_rows(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    const lpulite_mmio_row_t *values,
    size_t count
) {
    for (size_t index = 0; index < count; ++index) {
        write_row_unbarriered(
            dev,
            base_offset,
            first_row + (uint32_t)index,
            values[index]
        );
    }
    io_barrier();
}

void lpulite_read_rows(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    lpulite_mmio_row_t *values,
    size_t count
) {
    io_barrier();
    for (size_t index = 0; index < count; ++index) {
        values[index] = read_row_unbarriered(
            dev,
            base_offset,
            first_row + (uint32_t)index
        );
    }
    io_barrier();
}

void lpulite_fill_rows(
    lpulite_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    lpulite_mmio_row_t value,
    size_t count
) {
    for (size_t index = 0; index < count; ++index) {
        write_row_unbarriered(
            dev,
            base_offset,
            first_row + (uint32_t)index,
            value
        );
    }
    io_barrier();
}

void lpulite_copy_row(lpulite_mmio_t *dev, uint32_t src_base, uint32_t src_row, uint32_t dst_base, uint32_t dst_row) {
    lpulite_write_row(dev, dst_base, dst_row, lpulite_read_row(dev, src_base, src_row));
}

void lpulite_load_imem_page(lpulite_mmio_t *dev, const lpulite_mmio_row_t *program, size_t count) {
    if (count > LPULITE_IMEM_ROWS) {
        count = LPULITE_IMEM_ROWS;
    }
    lpulite_write_rows(dev, LPULITE_IMEM_OFFSET, 0u, program, count);
    if (count < LPULITE_IMEM_ROWS) {
        const lpulite_mmio_row_t zero = {0, 0, 0};
        lpulite_fill_rows(
            dev,
            LPULITE_IMEM_OFFSET,
            (uint32_t)count,
            zero,
            LPULITE_IMEM_ROWS - count
        );
    }
}

void lpulite_run_page(lpulite_mmio_t *dev, unsigned settle_us) {
    lpulite_write32(dev, LPULITE_CTRL_PC_LOAD, 0);
    lpulite_write32(dev, LPULITE_CTRL_RUN, 1);
    if (settle_us) {
        usleep(settle_us);
    }
    lpulite_write32(dev, LPULITE_CTRL_RUN, 0);
}

uint32_t lpulite_run_cycles_from(
    lpulite_mmio_t *dev,
    uint32_t pc,
    uint32_t cycles,
    unsigned poll_us
) {
    uint32_t before = lpulite_read32(dev, LPULITE_CTRL_CYCLES);
    lpulite_write32(dev, LPULITE_CTRL_PC_LOAD, pc);
    lpulite_write32(dev, LPULITE_CTRL_RUN_CYCLES, cycles);

    for (unsigned guard = 0; guard < 10000000u; ++guard) {
        uint32_t remaining = lpulite_read32(dev, LPULITE_CTRL_RUN_CYCLES);
        if (remaining == 0u) {
            break;
        }
        if (poll_us) {
            usleep(poll_us);
        }
    }
    uint32_t after = lpulite_read32(dev, LPULITE_CTRL_CYCLES);
    return after - before;
}

uint32_t lpulite_run_cycles(lpulite_mmio_t *dev, uint32_t cycles, unsigned poll_us) {
    return lpulite_run_cycles_from(dev, 0u, cycles, poll_us);
}
