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

int tinylpu_mmio_open(tinylpu_mmio_t *dev, uintptr_t phys_base, size_t span) {
    if (!dev) {
        errno = EINVAL;
        return -1;
    }

    dev->fd = -1;
    dev->span = span;
    dev->phys_base = phys_base;
    dev->base = NULL;

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
    io_barrier();
}

uint32_t tinylpu_read32(tinylpu_mmio_t *dev, uint32_t offset) {
    io_barrier();
    uint32_t value = *word_ptr(dev, offset);
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
    uint32_t addr = base_offset + row * TINYLPU_ROW_BYTES;
    tinylpu_write32(dev, addr + 0, value.w0);
    tinylpu_write32(dev, addr + 4, value.w1);
    /*
     * The current Avalon wrapper commits the 96-bit/72-bit assembled row when
     * word 2 is written, so word 2 must be last.
     */
    tinylpu_write32(dev, addr + 8, value.w2);
}

tinylpu_mmio_row_t tinylpu_read_row(tinylpu_mmio_t *dev, uint32_t base_offset, uint32_t row) {
    uint32_t addr = base_offset + row * TINYLPU_ROW_BYTES;
    /*
     * The wrapper has a registered SRAM read path. A same-address dummy read
     * gives the fabric one clocked MMIO transaction to present the row, matching
     * the existing System Console/JTAG bring-up code.
     */
    (void)tinylpu_read32(dev, addr);
    tinylpu_mmio_row_t value = {
        tinylpu_read32(dev, addr + 0),
        tinylpu_read32(dev, addr + 4),
        tinylpu_read32(dev, addr + 8),
    };
    return value;
}

void tinylpu_copy_row(tinylpu_mmio_t *dev, uint32_t src_base, uint32_t src_row, uint32_t dst_base, uint32_t dst_row) {
    tinylpu_write_row(dev, dst_base, dst_row, tinylpu_read_row(dev, src_base, src_row));
}

void tinylpu_load_imem_page(tinylpu_mmio_t *dev, const tinylpu_mmio_row_t *program, size_t count) {
    for (size_t row = 0; row < TINYLPU_IMEM_ROWS; ++row) {
        tinylpu_mmio_row_t value = {0, 0, 0};
        if (row < count) {
            value = program[row];
        }
        tinylpu_write_row(dev, TINYLPU_IMEM_OFFSET, (uint32_t)row, value);
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

    for (;;) {
        uint32_t run = tinylpu_read32(dev, TINYLPU_CTRL_RUN);
        uint32_t remaining = tinylpu_read32(dev, TINYLPU_CTRL_RUN_CYCLES);
        if ((run == 0u) && (remaining == 0u)) {
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
