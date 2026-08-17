#include "tinylpu_hps_mmio.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static volatile uint32_t *word_ptr(tinylpu_mmio_t *dev, uint32_t offset) {
    return (volatile uint32_t *)(void *)(dev->base + offset);
}

static void io_barrier(void) {
    __sync_synchronize();
}

static void write_row_unbarriered(
    tinylpu_mmio_t *dev,
    uint32_t base_offset,
    uint32_t row,
    tinylpu_mmio_row_t value
) {
    uint32_t addr = base_offset + row * TINYLPU_ROW_BYTES;
    *word_ptr(dev, addr + 0u) = value.w0;
    *word_ptr(dev, addr + 4u) = value.w1;
    /* Word 2 commits the wrapper's assembled row and must remain last. */
    *word_ptr(dev, addr + 8u) = value.w2;
    dev->stats.write32 += 3u;
    dev->stats.row_writes += 1u;
    if (base_offset == TINYLPU_IMEM_OFFSET) {
        dev->stats.imem_row_writes += 1u;
    }
}

static tinylpu_mmio_row_t read_row_unbarriered(
    tinylpu_mmio_t *dev,
    uint32_t base_offset,
    uint32_t row
) {
    uint32_t addr = base_offset + row * TINYLPU_ROW_BYTES;
    /* Prime the wrapper's registered SRAM read path. */
    (void)*word_ptr(dev, addr);
    tinylpu_mmio_row_t value = {
        *word_ptr(dev, addr + 0u),
        *word_ptr(dev, addr + 4u),
        *word_ptr(dev, addr + 8u),
    };
    dev->stats.read32 += 4u;
    dev->stats.row_reads += 1u;
    return value;
}

int tinylpu_mmio_open(tinylpu_mmio_t *dev, uintptr_t phys_base, size_t span) {
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

tinylpu_mmio_stats_t tinylpu_mmio_get_stats(const tinylpu_mmio_t *dev) {
    return dev->stats;
}

void tinylpu_mmio_close(tinylpu_mmio_t *dev) {
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

void tinylpu_write32(tinylpu_mmio_t *dev, uint32_t offset, uint32_t value) {
    *word_ptr(dev, offset) = value;
    dev->stats.write32 += 1u;
    io_barrier();
}

uint32_t tinylpu_read32(tinylpu_mmio_t *dev, uint32_t offset) {
    io_barrier();
    uint32_t value = *word_ptr(dev, offset);
    dev->stats.read32 += 1u;
    io_barrier();
    return value;
}

void tinylpu_soft_reset(tinylpu_mmio_t *dev, uint32_t cycles, unsigned poll_us) {
    if (cycles == 0u) {
        cycles = 16u;
    }
    tinylpu_write32(dev, TINYLPU_CTRL_RUN, 0u);
    tinylpu_write32(dev, TINYLPU_CTRL_RUN_CYCLES, 0u);
    tinylpu_write32(dev, TINYLPU_CTRL_PC_LOAD, 0u);
    tinylpu_write32(dev, TINYLPU_CTRL_SOFT_RESET, cycles);
    for (unsigned guard = 0; guard < 100000u; ++guard) {
        uint32_t remaining = tinylpu_read32(dev, TINYLPU_CTRL_SOFT_RESET);
        if (remaining == 0u) {
            break;
        }
        if (poll_us) {
            usleep(poll_us);
        }
    }
}

void tinylpu_write_row(tinylpu_mmio_t *dev, uint32_t base_offset, uint32_t row, tinylpu_mmio_row_t value) {
    write_row_unbarriered(dev, base_offset, row, value);
    io_barrier();
}

tinylpu_mmio_row_t tinylpu_read_row(tinylpu_mmio_t *dev, uint32_t base_offset, uint32_t row) {
    io_barrier();
    tinylpu_mmio_row_t value = read_row_unbarriered(dev, base_offset, row);
    io_barrier();
    return value;
}

void tinylpu_write_rows(
    tinylpu_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    const tinylpu_mmio_row_t *values,
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

void tinylpu_read_rows(
    tinylpu_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    tinylpu_mmio_row_t *values,
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

void tinylpu_fill_rows(
    tinylpu_mmio_t *dev,
    uint32_t base_offset,
    uint32_t first_row,
    tinylpu_mmio_row_t value,
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

void tinylpu_copy_row(tinylpu_mmio_t *dev, uint32_t src_base, uint32_t src_row, uint32_t dst_base, uint32_t dst_row) {
    tinylpu_write_row(dev, dst_base, dst_row, tinylpu_read_row(dev, src_base, src_row));
}

void tinylpu_load_imem_page(tinylpu_mmio_t *dev, const tinylpu_mmio_row_t *program, size_t count) {
    if (count > TINYLPU_IMEM_ROWS) {
        count = TINYLPU_IMEM_ROWS;
    }
    tinylpu_write_rows(dev, TINYLPU_IMEM_OFFSET, 0u, program, count);
    if (count < TINYLPU_IMEM_ROWS) {
        const tinylpu_mmio_row_t zero = {0, 0, 0};
        tinylpu_fill_rows(
            dev,
            TINYLPU_IMEM_OFFSET,
            (uint32_t)count,
            zero,
            TINYLPU_IMEM_ROWS - count
        );
    }
}

void tinylpu_run_page(tinylpu_mmio_t *dev, unsigned settle_us) {
    tinylpu_write32(dev, TINYLPU_CTRL_PC_LOAD, 0);
    tinylpu_write32(dev, TINYLPU_CTRL_RUN, 1);
    if (settle_us) {
        usleep(settle_us);
    }
    tinylpu_write32(dev, TINYLPU_CTRL_RUN, 0);
}

uint32_t tinylpu_run_cycles_from(
    tinylpu_mmio_t *dev,
    uint32_t pc,
    uint32_t cycles,
    unsigned poll_us
) {
    uint32_t before = tinylpu_read32(dev, TINYLPU_CTRL_CYCLES);
    tinylpu_write32(dev, TINYLPU_CTRL_PC_LOAD, pc);
    tinylpu_write32(dev, TINYLPU_CTRL_RUN_CYCLES, cycles);

    for (unsigned guard = 0; guard < 10000000u; ++guard) {
        uint32_t remaining = tinylpu_read32(dev, TINYLPU_CTRL_RUN_CYCLES);
        if (remaining == 0u) {
            break;
        }
        if (poll_us) {
            usleep(poll_us);
        }
    }
    uint32_t after = tinylpu_read32(dev, TINYLPU_CTRL_CYCLES);
    return after - before;
}

uint32_t tinylpu_run_cycles(tinylpu_mmio_t *dev, uint32_t cycles, unsigned poll_us) {
    return tinylpu_run_cycles_from(dev, 0u, cycles, poll_us);
}
