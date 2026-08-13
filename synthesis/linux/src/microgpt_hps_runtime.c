#include "tinylpu_hps_mmio.h"
#include "microgpt_hps_image.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    ATTENTION_CURRENT = 0,
    ATTENTION_HOST = 1,
    ATTENTION_FPGA_SOFTMAX = 2,
    ATTENTION_FPGA_MXM = 3,
} attention_mode_t;

typedef enum {
    DECODE_TARGET = 0,
    DECODE_GREEDY = 1,
} decode_mode_t;

typedef struct {
    uintptr_t base;
    size_t span;
    unsigned settle_us;
    bool skip_load_weights;
    bool verbose;
    bool probe_only;
    bool sxm_probe;
    bool benchmark;
    bool host_broadcasts;
    attention_mode_t attention_mode;
    decode_mode_t decode_mode;
    unsigned max_new_tokens;
} runtime_options_t;

static double monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0.0;
    }
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static void print_benchmark(
    int prompt_tokens,
    unsigned output_tokens,
    unsigned prefill_steps,
    unsigned decode_steps,
    double request_seconds,
    double prefill_seconds,
    double decode_step_seconds,
    double ttft_seconds
) {
    double e2e_tps = request_seconds > 0.0
        ? (double)output_tokens / request_seconds
        : 0.0;
    double prefill_sps = prefill_seconds > 0.0
        ? (double)prefill_steps / prefill_seconds
        : 0.0;
    double decode_sps = decode_step_seconds > 0.0
        ? (double)decode_steps / decode_step_seconds
        : 0.0;

    fprintf(stderr,
        "[perf] ARM+FPGA wall clock (reset/MMIO/page loads/polling/attention/decode included)\n"
        "[perf] prompt_tokens=%d output_tokens=%u lpu_steps=%u (prefill=%u decode=%u)\n"
        "[perf] request=%.3f s  TTFT=%s%.3f s  end_to_end=%.3f output tokens/s\n"
        "[perf] prefill=%.3f s (%.3f LPU steps/s)  decode_steps=%.3f s (%.3f LPU steps/s)\n",
        prompt_tokens,
        output_tokens,
        prefill_steps + decode_steps,
        prefill_steps,
        decode_steps,
        request_seconds,
        output_tokens ? "" : "n/a ",
        output_tokens ? ttft_seconds : 0.0,
        e2e_tps,
        prefill_seconds,
        prefill_sps,
        decode_step_seconds,
        decode_sps);
    fflush(stderr);
}

static char token_char(int token_id);
static bool is_target_name(const char *text);
static bool target_has_prefix(const char *text);

static tinylpu_mmio_row_t as_mmio_row(tinylpu_row96_t row) {
    tinylpu_mmio_row_t out = {row.w0, row.w1, row.w2};
    return out;
}

static int8_t s8(uint32_t value) {
    uint8_t byte = (uint8_t)(value & 0xFFu);
    return (int8_t)byte;
}

static void unpack_row(tinylpu_mmio_row_t row, int8_t lanes[MICROGPT_LANES], int8_t *scale) {
    uint64_t packed = (uint64_t)row.w0 | ((uint64_t)row.w1 << 32);
    for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
        lanes[lane] = s8((uint32_t)(packed >> (lane * 8)));
    }
    *scale = s8(row.w2);
}

static void row_to_vector(const tinylpu_mmio_row_t rows[MICROGPT_ROWS_PER_VEC], double vec[MICROGPT_N_EMBD]) {
    for (int r = 0; r < MICROGPT_ROWS_PER_VEC; ++r) {
        int8_t lanes[MICROGPT_LANES];
        int8_t scale;
        unpack_row(rows[r], lanes, &scale);
        for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
            vec[r * MICROGPT_LANES + lane] = ldexp((double)lanes[lane], scale);
        }
    }
}

static tinylpu_mmio_row_t pack_float_row(const double *values, int count) {
    double absmax = 0.0;
    for (int i = 0; i < count; ++i) {
        double v = fabs(values[i]);
        if (v > absmax) {
            absmax = v;
        }
    }

    int scale = 0;
    if (absmax > 0.0) {
        scale = (int)ceil(log2(absmax / 127.0));
        if (scale < -128) scale = -128;
        if (scale > 127) scale = 127;
    }

    double inv = ldexp(1.0, -scale);
    uint64_t packed = 0;
    for (int i = 0; i < MICROGPT_LANES; ++i) {
        int q = 0;
        if (i < count) {
            q = (int)llround(values[i] * inv);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
        }
        packed |= ((uint64_t)((uint8_t)((int8_t)q))) << (i * 8);
    }

    tinylpu_mmio_row_t row = {
        (uint32_t)(packed & 0xFFFFFFFFu),
        (uint32_t)(packed >> 32),
        (uint32_t)((uint8_t)((int8_t)scale)),
    };
    return row;
}

static tinylpu_mmio_row_t pack_float_row_at_scale(
    const double *values,
    int count,
    int scale
) {
    if (scale < -128) scale = -128;
    if (scale > 127) scale = 127;
    const double inv = ldexp(1.0, -scale);
    uint64_t packed = 0;
    for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
        int q = 0;
        if (lane < count) {
            q = (int)llround(values[lane] * inv);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
        }
        packed |= ((uint64_t)((uint8_t)((int8_t)q))) << (lane * 8);
    }
    tinylpu_mmio_row_t row = {
        (uint32_t)(packed & 0xFFFFFFFFu),
        (uint32_t)(packed >> 32),
        (uint32_t)((uint8_t)((int8_t)scale)),
    };
    return row;
}

static tinylpu_mmio_row_t pack_quant_row(const int8_t lanes[MICROGPT_LANES], int8_t scale) {
    uint64_t packed = 0;
    for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
        packed |= ((uint64_t)((uint8_t)lanes[lane])) << (lane * 8);
    }
    tinylpu_mmio_row_t row = {
        (uint32_t)(packed & 0xFFFFFFFFu),
        (uint32_t)(packed >> 32),
        (uint32_t)((uint8_t)scale),
    };
    return row;
}

static void stage_broadcast_row(tinylpu_mmio_t *dev, uint32_t src_row, uint32_t dst_base) {
    int8_t lanes[MICROGPT_LANES];
    int8_t scale;
    unpack_row(tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, src_row), lanes, &scale);
    for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
        int8_t replicated[MICROGPT_LANES];
        for (int out = 0; out < MICROGPT_LANES; ++out) {
            replicated[out] = lanes[lane];
        }
        tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, dst_base + (uint32_t)lane, pack_quant_row(replicated, scale));
    }
}

static void stage_broadcast_pair(tinylpu_mmio_t *dev, uint32_t src_row0, uint32_t src_row1, uint32_t dst_base) {
    stage_broadcast_row(dev, src_row0, dst_base);
    stage_broadcast_row(dev, src_row1, dst_base + MICROGPT_LANES);
}

static void vector_to_rows(const double vec[MICROGPT_N_EMBD], tinylpu_mmio_row_t rows[MICROGPT_ROWS_PER_VEC]) {
    for (int r = 0; r < MICROGPT_ROWS_PER_VEC; ++r) {
        rows[r] = pack_float_row(&vec[r * MICROGPT_LANES], MICROGPT_LANES);
    }
}

static void softmax(const double *scores, int count, double *weights) {
    double max_v = scores[0];
    for (int i = 1; i < count; ++i) {
        if (scores[i] > max_v) max_v = scores[i];
    }
    double total = 0.0;
    for (int i = 0; i < count; ++i) {
        weights[i] = exp(scores[i] - max_v);
        total += weights[i];
    }
    if (total == 0.0) {
        for (int i = 0; i < count; ++i) weights[i] = 1.0 / (double)count;
    } else {
        for (int i = 0; i < count; ++i) weights[i] /= total;
    }
}

static void load_mem1(tinylpu_mmio_t *dev, bool verbose) {
    for (uint32_t row = 0; row < MICROGPT_MEM1_ROWS; ++row) {
        if (verbose && (row % 64u) == 0u) {
            fprintf(stderr, "[mem1] row %u/%u\n", row, MICROGPT_MEM1_ROWS);
            fflush(stderr);
        }
        tinylpu_write_row(dev, TINYLPU_MEM1_OFFSET, row, as_mmio_row(g_microgpt_mem1[row]));
    }
    if (verbose) {
        fprintf(stderr, "[mem1] row %u/%u\n", MICROGPT_MEM1_ROWS, MICROGPT_MEM1_ROWS);
        fflush(stderr);
    }
}

static void clear_runtime_state(tinylpu_mmio_t *dev, bool verbose) {
    tinylpu_mmio_row_t zero = {0, 0, 0};
    if (verbose) {
        fprintf(stderr, "[reset] clear MEM0 scratch/K cache and MEM1 V cache\n");
        fflush(stderr);
    }
    for (uint32_t row = 0; row <= MICROGPT_MEM0_LOGIT_ROW3; ++row) {
        tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, row, zero);
    }
    for (uint32_t row = 0; row < MICROGPT_BLOCK_SIZE * MICROGPT_ROWS_PER_VEC; ++row) {
        tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_K_CACHE_BASE + row, zero);
        tinylpu_write_row(dev, TINYLPU_MEM1_OFFSET, MICROGPT_V_CACHE_BASE + row, zero);
    }
}

static void reset_prompt_state(tinylpu_mmio_t *dev, const runtime_options_t *opt) {
    if (opt->verbose) {
        fprintf(stderr, "[reset] assert LPU soft reset\n");
        fflush(stderr);
    }
    tinylpu_soft_reset(dev, 32u, opt->settle_us);
    clear_runtime_state(dev, opt->verbose);
    tinylpu_soft_reset(dev, 32u, opt->settle_us);
}

static void debug_dump_row(tinylpu_mmio_t *dev, const char *name, uint32_t base, uint32_t row) {
    tinylpu_mmio_row_t value = tinylpu_read_row(dev, base, row);
    fprintf(stderr, "[row] %-10s %s[%u]=%08x %08x %08x\n",
            name,
            base == TINYLPU_MEM1_OFFSET ? "MEM1" : "MEM0",
            row,
            value.w0,
            value.w1,
            value.w2);
}

static void debug_dump_runtime_rows(tinylpu_mmio_t *dev, const char *stage) {
    fprintf(stderr, "[debug] %s row snapshot\n", stage);
    debug_dump_row(dev, "token0", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_TOKEN_ROW0);
    debug_dump_row(dev, "token1", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_TOKEN_ROW1);
    debug_dump_row(dev, "pos0", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_POS_ROW0);
    debug_dump_row(dev, "pos1", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_POS_ROW1);
    debug_dump_row(dev, "embed0", TINYLPU_MEM0_OFFSET, 8u);
    debug_dump_row(dev, "embed1", TINYLPU_MEM0_OFFSET, 9u);
    debug_dump_row(dev, "q0", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_Q_ROW0);
    debug_dump_row(dev, "q1", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_Q_ROW1);
    debug_dump_row(dev, "k0", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_K_ROW0);
    debug_dump_row(dev, "k1", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_K_ROW1);
    debug_dump_row(dev, "v0", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_V_ROW0);
    debug_dump_row(dev, "v1", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_V_ROW1);
    debug_dump_row(dev, "attn0", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_ROW0);
    debug_dump_row(dev, "attn1", TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_ROW1);
    fflush(stderr);
}

static void run_image(
    tinylpu_mmio_t *dev,
    const char *label,
    const tinylpu_row96_t *image,
    size_t start_pc,
    size_t instruction_count,
    const runtime_options_t *opt
) {
    size_t done = 0;
    unsigned page = 0;
    while (done < instruction_count) {
        size_t remain = instruction_count - done;
        size_t page_count = remain < MICROGPT_IMEM_PAGE_SIZE ? remain : MICROGPT_IMEM_PAGE_SIZE;
        ++page;
        if (opt->verbose) {
            fprintf(stderr, "[%s] load page %u: pc=%u count=%u\n",
                    label,
                    page,
                    (unsigned)(start_pc + done),
                    (unsigned)page_count);
            fflush(stderr);
        }
        // Exact-cycle execution cannot reach the rest of IMEM.  Write only
        // the active page and its eight retirement NOPs instead of clearing
        // all 1024 rows on every page load.
        size_t rows_to_write = page_count + 8u;
        if (rows_to_write > TINYLPU_IMEM_ROWS) {
            rows_to_write = TINYLPU_IMEM_ROWS;
        }
        for (size_t row = 0; row < rows_to_write; ++row) {
            tinylpu_mmio_row_t value = {0, 0, 0};
            if (row < page_count) {
                value = as_mmio_row(image[start_pc + done + row]);
            }
            tinylpu_write_row(dev, TINYLPU_IMEM_OFFSET, (uint32_t)row, value);
        }
        if (opt->verbose) {
            fprintf(stderr, "[%s] run page %u\n", label, page);
            fflush(stderr);
        }
        uint32_t requested_cycles = (uint32_t)(page_count + 8u);
        uint32_t actual_cycles = tinylpu_run_cycles(dev, requested_cycles, opt->settle_us);
        if (opt->verbose) {
            fprintf(stderr, "[%s] done page %u cycles=%u/%u\n",
                    label,
                    page,
                    actual_cycles,
                    requested_cycles);
            fflush(stderr);
        }
        done += page_count;
    }
}

static void run_program(tinylpu_mmio_t *dev, const char *label, size_t start_pc, size_t instruction_count, const runtime_options_t *opt) {
    run_image(dev, label, g_microgpt_vliw, start_pc, instruction_count, opt);
}

static void run_softmax_program(
    tinylpu_mmio_t *dev,
    const runtime_options_t *opt,
    bool load_image
) {
    if (load_image) {
        run_image(
            dev,
            "attention-softmax",
            g_microgpt_softmax_vliw,
            0,
            MICROGPT_SOFTMAX_INSTRUCTIONS,
            opt
        );
        return;
    }

    // All four heads use the same one-page kernel, with no other LPU program
    // running between them.  Keep it resident after head 0 instead of writing
    // all 1024 IMEM rows over MMIO three more times per token.
    uint32_t requested_cycles = MICROGPT_SOFTMAX_INSTRUCTIONS + 8u;
    if (opt->verbose) {
        fprintf(stderr, "[attention-softmax] run resident kernel cycles=%u\n",
                requested_cycles);
        fflush(stderr);
    }
    uint32_t actual_cycles = tinylpu_run_cycles(dev, requested_cycles, opt->settle_us);
    if (opt->verbose) {
        fprintf(stderr, "[attention-softmax] resident kernel done cycles=%u/%u\n",
                actual_cycles, requested_cycles);
        fflush(stderr);
    }
}

static void write_step_inputs(tinylpu_mmio_t *dev, int token_id, int pos_id, bool verbose) {
    if (verbose) {
        fprintf(stderr, "[input] token=%d pos=%d\n", token_id, pos_id);
        fflush(stderr);
    }
    uint32_t wte = MICROGPT_MEM1_WTE_BASE + (uint32_t)token_id * MICROGPT_ROWS_PER_VEC;
    uint32_t wpe = MICROGPT_MEM1_WPE_BASE + (uint32_t)pos_id * MICROGPT_ROWS_PER_VEC;
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_TOKEN_ROW0, as_mmio_row(g_microgpt_mem1[wte + 0]));
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_TOKEN_ROW1, as_mmio_row(g_microgpt_mem1[wte + 1]));
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_POS_ROW0, as_mmio_row(g_microgpt_mem1[wpe + 0]));
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_POS_ROW1, as_mmio_row(g_microgpt_mem1[wpe + 1]));
}

static void cache_current_kv(tinylpu_mmio_t *dev, int pos_id, bool verbose) {
    if (verbose) {
        fprintf(stderr, "[kv] cache current pos=%d\n", pos_id);
        fflush(stderr);
    }
    uint32_t k_base = MICROGPT_K_CACHE_BASE + (uint32_t)pos_id * MICROGPT_ROWS_PER_VEC;
    uint32_t v_base = MICROGPT_V_CACHE_BASE + (uint32_t)pos_id * MICROGPT_ROWS_PER_VEC;
    tinylpu_copy_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_K_ROW0, TINYLPU_MEM0_OFFSET, k_base + 0);
    tinylpu_copy_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_K_ROW1, TINYLPU_MEM0_OFFSET, k_base + 1);
    tinylpu_copy_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_V_ROW0, TINYLPU_MEM1_OFFSET, v_base + 0);
    tinylpu_copy_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_V_ROW1, TINYLPU_MEM1_OFFSET, v_base + 1);
}

static void stage_current_attention(tinylpu_mmio_t *dev, bool verbose) {
    if (verbose) {
        fprintf(stderr, "[attention] stage current V\n");
        fflush(stderr);
    }
    tinylpu_copy_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_V_ROW0, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_ROW0);
    tinylpu_copy_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_V_ROW1, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_ROW1);
}

static void fpga_softmax(
    tinylpu_mmio_t *dev,
    const double scores[MICROGPT_BLOCK_SIZE],
    int count,
    double weights[MICROGPT_BLOCK_SIZE],
    const runtime_options_t *opt,
    bool load_program
) {
    int8_t masked_lanes[MICROGPT_LANES];
    for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
        masked_lanes[lane] = -127;
    }
    for (int pos = 0; pos < MICROGPT_SOFTMAX_CHUNKS; ++pos) {
        tinylpu_mmio_row_t row;
        if (pos < count) {
            double duplicated[MICROGPT_LANES];
            for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
                duplicated[lane] = scores[pos];
            }
            row = pack_float_row(duplicated, MICROGPT_LANES);
        } else {
            // Software supplies the causal/dynamic-length mask; the FPGA does
            // every exp, reciprocal, and normalization operation.
            row = pack_quant_row(masked_lanes, 0);
        }
        tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_SOFTMAX_IN_BASE + (uint32_t)pos, row);
    }

    run_softmax_program(dev, opt, load_program);
    for (int pos = 0; pos < count; ++pos) {
        int8_t lanes[MICROGPT_LANES];
        int8_t scale;
        unpack_row(
            tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_SOFTMAX_OUT_BASE + (uint32_t)pos),
            lanes,
            &scale
        );
        double probability = 0.0;
        for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
            probability += ldexp((double)(uint8_t)lanes[lane], scale);
        }
        weights[pos] = probability;
    }
}

static void stage_host_attention(
    tinylpu_mmio_t *dev,
    int through_pos,
    const runtime_options_t *opt,
    bool use_fpga_softmax
) {
    if (opt->verbose) {
        fprintf(stderr, "[attention] ARM QK/PV, %s softmax, through_pos=%d\n",
                use_fpga_softmax ? "FPGA" : "ARM", through_pos);
        fflush(stderr);
    }
    tinylpu_mmio_row_t q_rows[MICROGPT_ROWS_PER_VEC] = {
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_Q_ROW0),
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_Q_ROW1),
    };
    double q[MICROGPT_N_EMBD];
    row_to_vector(q_rows, q);

    double context[MICROGPT_N_EMBD] = {0};
    double scores[MICROGPT_BLOCK_SIZE];
    double weights[MICROGPT_BLOCK_SIZE];

    for (int head = 0; head < MICROGPT_N_HEAD; ++head) {
        int base = head * MICROGPT_HEAD_DIM;
        for (int pos = 0; pos <= through_pos; ++pos) {
            tinylpu_mmio_row_t k_rows[MICROGPT_ROWS_PER_VEC] = {
                tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_K_CACHE_BASE + pos * MICROGPT_ROWS_PER_VEC + 0),
                tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_K_CACHE_BASE + pos * MICROGPT_ROWS_PER_VEC + 1),
            };
            double key[MICROGPT_N_EMBD];
            row_to_vector(k_rows, key);
            double dot = 0.0;
            for (int i = 0; i < MICROGPT_HEAD_DIM; ++i) {
                dot += q[base + i] * key[base + i];
            }
            scores[pos] = dot / sqrt((double)MICROGPT_HEAD_DIM);
        }
        if (use_fpga_softmax) {
            fpga_softmax(dev, scores, through_pos + 1, weights, opt, head == 0);
        } else {
            softmax(scores, through_pos + 1, weights);
        }
        for (int i = 0; i < MICROGPT_HEAD_DIM; ++i) {
            double acc = 0.0;
            for (int pos = 0; pos <= through_pos; ++pos) {
                tinylpu_mmio_row_t v_rows[MICROGPT_ROWS_PER_VEC] = {
                    tinylpu_read_row(dev, TINYLPU_MEM1_OFFSET, MICROGPT_V_CACHE_BASE + pos * MICROGPT_ROWS_PER_VEC + 0),
                    tinylpu_read_row(dev, TINYLPU_MEM1_OFFSET, MICROGPT_V_CACHE_BASE + pos * MICROGPT_ROWS_PER_VEC + 1),
                };
                double value[MICROGPT_N_EMBD];
                row_to_vector(v_rows, value);
                acc += weights[pos] * value[base + i];
            }
            context[base + i] = acc;
        }
    }

    tinylpu_mmio_row_t context_rows[MICROGPT_ROWS_PER_VEC];
    vector_to_rows(context, context_rows);
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_ROW0, context_rows[0]);
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_ROW1, context_rows[1]);
}

static void load_attention_image(tinylpu_mmio_t *dev, const runtime_options_t *opt) {
    if (opt->verbose) {
        fprintf(stderr, "[attention-mxm] load resident image: %u instructions\n",
                MICROGPT_ATTENTION_INSTRUCTIONS);
        fflush(stderr);
    }
    for (uint32_t row = 0; row < MICROGPT_ATTENTION_INSTRUCTIONS; ++row) {
        tinylpu_write_row(
            dev,
            TINYLPU_IMEM_OFFSET,
            row,
            as_mmio_row(g_microgpt_attention_vliw[row])
        );
    }
    // Park ICU on the final stopped-PC guard. Without this, loading row 0 can
    // leave the K-tile MEM0 read asserted while ARM is staging operands.
    tinylpu_write32(
        dev,
        TINYLPU_CTRL_PC_LOAD,
        MICROGPT_ATTENTION_INSTRUCTIONS - 1u
    );
}

static void run_attention_section(
    tinylpu_mmio_t *dev,
    const char *label,
    uint32_t start,
    uint32_t instructions,
    const runtime_options_t *opt
) {
    if (opt->verbose) {
        fprintf(stderr, "[attention-mxm] run %s pc=%u cycles=%u\n",
                label, start, instructions);
        fflush(stderr);
    }
    uint32_t actual = tinylpu_run_cycles_from(
        dev,
        start,
        instructions,
        opt->settle_us
    );
    if (actual != instructions) {
        fprintf(stderr,
                "[attention-mxm] warning: %s cycle mismatch actual=%u requested=%u\n",
                label, actual, instructions);
        fflush(stderr);
    }
}

static int8_t clamp_scale(int scale) {
    if (scale < -128) return -128;
    if (scale > 127) return 127;
    return (int8_t)scale;
}

static void stage_mxm_attention(
    tinylpu_mmio_t *dev,
    int through_pos,
    const runtime_options_t *opt
) {
    if (opt->verbose) {
        fprintf(stderr,
                "[attention] FPGA SXM K^T -> FPGA MXM QK -> FPGA softmax -> FPGA MXM PV, through_pos=%d\n",
                through_pos);
        fflush(stderr);
    }
    load_attention_image(dev, opt);

    tinylpu_mmio_row_t q_rows[MICROGPT_ROWS_PER_VEC] = {
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_Q_ROW0),
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_Q_ROW1),
    };
    int8_t q_lanes[MICROGPT_ROWS_PER_VEC][MICROGPT_LANES];
    int8_t q_scales[MICROGPT_ROWS_PER_VEC];
    for (int row = 0; row < MICROGPT_ROWS_PER_VEC; ++row) {
        unpack_row(q_rows[row], q_lanes[row], &q_scales[row]);
    }

    double keys[MICROGPT_BLOCK_SIZE][MICROGPT_N_EMBD] = {{0}};
    tinylpu_mmio_row_t value_rows[MICROGPT_BLOCK_SIZE][MICROGPT_ROWS_PER_VEC];
    memset(value_rows, 0, sizeof(value_rows));
    for (int pos = 0; pos <= through_pos; ++pos) {
        tinylpu_mmio_row_t k_rows[MICROGPT_ROWS_PER_VEC] = {
            tinylpu_read_row(
                dev,
                TINYLPU_MEM0_OFFSET,
                MICROGPT_K_CACHE_BASE + (uint32_t)pos * MICROGPT_ROWS_PER_VEC + 0u
            ),
            tinylpu_read_row(
                dev,
                TINYLPU_MEM0_OFFSET,
                MICROGPT_K_CACHE_BASE + (uint32_t)pos * MICROGPT_ROWS_PER_VEC + 1u
            ),
        };
        row_to_vector(k_rows, keys[pos]);
        for (int row = 0; row < MICROGPT_ROWS_PER_VEC; ++row) {
            value_rows[pos][row] = tinylpu_read_row(
                dev,
                TINYLPU_MEM1_OFFSET,
                MICROGPT_V_CACHE_BASE + (uint32_t)pos * MICROGPT_ROWS_PER_VEC + (uint32_t)row
            );
        }
    }

    int8_t masked_lanes[MICROGPT_LANES];
    int8_t zero_lanes[MICROGPT_LANES] = {0};
    for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
        masked_lanes[lane] = -127;
    }
    const tinylpu_mmio_row_t masked_row = pack_quant_row(masked_lanes, 0);
    const tinylpu_mmio_row_t zero_row = pack_quant_row(zero_lanes, 0);

    for (int head = 0; head < MICROGPT_N_HEAD; ++head) {
        const int row_index = head / 2;
        const int lane_base = (head % 2) * MICROGPT_HEAD_DIM;

        // Replicate the four quantized Q scalars into MXM input rows. This and
        // the cache scale alignment below are layout/representation staging.
        for (int dim = 0; dim < MICROGPT_HEAD_DIM; ++dim) {
            int8_t replicated[MICROGPT_LANES];
            for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
                replicated[lane] = q_lanes[row_index][lane_base + dim];
            }
            tinylpu_write_row(
                dev,
                TINYLPU_MEM0_OFFSET,
                MICROGPT_MEM0_ATTN_Q_BCAST_BASE + (uint32_t)dim,
                pack_quant_row(replicated, q_scales[row_index])
            );
        }

        // SXM performs the actual position-by-dimension transpose for each
        // 8-token K tile. Its tile carries one shared exponent, so ARM first
        // aligns the heterogeneous cached-row exponents and masks the lanes
        // outside this head. This is representation/layout work only.
        for (int block = 0; block < 2; ++block) {
            if (block * MICROGPT_LANES > through_pos) {
                for (int dim = 0; dim < MICROGPT_HEAD_DIM; ++dim) {
                    tinylpu_write_row(
                        dev,
                        TINYLPU_MEM1_OFFSET,
                        MICROGPT_MEM1_ATTN_KT_STAGE_BASE + (uint32_t)(dim * 2 + block),
                        zero_row
                    );
                }
                continue;
            }
            double tile[MICROGPT_LANES][MICROGPT_LANES] = {{0}};
            double absmax = 0.0;
            for (int tile_pos = 0; tile_pos < MICROGPT_LANES; ++tile_pos) {
                const int pos = block * MICROGPT_LANES + tile_pos;
                if (pos <= through_pos) {
                    for (int dim = 0; dim < MICROGPT_HEAD_DIM; ++dim) {
                        const double value = keys[pos][head * MICROGPT_HEAD_DIM + dim];
                        // All heads use columns 0..3. The block-specific SXM
                        // entry point sends these four transposed rows through
                        // the LPU's VXM store adapter into interleaved MEM1.
                        tile[tile_pos][dim] = value;
                        if (fabs(value) > absmax) absmax = fabs(value);
                    }
                }
            }
            int tile_scale = 0;
            if (absmax > 0.0) {
                tile_scale = (int)ceil(log2(absmax / 127.0));
                if (tile_scale < -128) tile_scale = -128;
                if (tile_scale > 127) tile_scale = 127;
            }
            for (int tile_pos = 0; tile_pos < MICROGPT_LANES; ++tile_pos) {
                tinylpu_write_row(
                    dev,
                    TINYLPU_MEM0_OFFSET,
                    MICROGPT_MEM0_ATTN_K_TILE_IN_BASE + (uint32_t)tile_pos,
                    pack_float_row_at_scale(
                        tile[tile_pos],
                        MICROGPT_LANES,
                        tile_scale
                    )
                );
            }
            if (block == 0) {
                run_attention_section(
                    dev,
                    "K[0:8] transpose on SXM -> MEM1",
                    MICROGPT_ATTN_K_TRANSPOSE_BLOCK0_START,
                    MICROGPT_ATTN_K_TRANSPOSE_BLOCK0_INSTRUCTIONS,
                    opt
                );
            } else {
                run_attention_section(
                    dev,
                    "K[8:16] transpose on SXM -> MEM1",
                    MICROGPT_ATTN_K_TRANSPOSE_BLOCK1_START,
                    MICROGPT_ATTN_K_TRANSPOSE_BLOCK1_INSTRUCTIONS,
                    opt
                );
            }
        }

        run_attention_section(
            dev,
            "QK",
            MICROGPT_ATTN_QK_START,
            MICROGPT_ATTN_QK_INSTRUCTIONS,
            opt
        );

        tinylpu_mmio_row_t score_rows[2] = {
            tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_QK_SCORE_BASE + 0u),
            tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_ATTN_QK_SCORE_BASE + 1u),
        };
        int8_t score_lanes[2][MICROGPT_LANES];
        int8_t score_scales[2];
        for (int block = 0; block < 2; ++block) {
            unpack_row(score_rows[block], score_lanes[block], &score_scales[block]);
        }
        for (int pos = 0; pos < MICROGPT_BLOCK_SIZE; ++pos) {
            tinylpu_mmio_row_t row = masked_row;
            if (pos <= through_pos) {
                const int block = pos / MICROGPT_LANES;
                int8_t duplicated[MICROGPT_LANES];
                for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
                    duplicated[lane] = score_lanes[block][pos % MICROGPT_LANES];
                }
                // head_dim=4, therefore QK/sqrt(head_dim) is exactly a one-bit
                // exponent decrement and requires no ARM floating-point math.
                row = pack_quant_row(
                    duplicated,
                    clamp_scale((int)score_scales[block] - 1)
                );
            }
            tinylpu_write_row(
                dev,
                TINYLPU_MEM0_OFFSET,
                MICROGPT_SOFTMAX_IN_BASE + (uint32_t)pos,
                row
            );
        }

        run_attention_section(
            dev,
            "softmax",
            MICROGPT_ATTN_SOFTMAX_START,
            MICROGPT_ATTN_SOFTMAX_INSTRUCTIONS,
            opt
        );

        for (int pos = 0; pos < MICROGPT_BLOCK_SIZE; ++pos) {
            tinylpu_mmio_row_t probability = zero_row;
            tinylpu_mmio_row_t value = zero_row;
            if (pos <= through_pos) {
                int8_t probability_lanes[MICROGPT_LANES];
                int8_t probability_scale;
                unpack_row(
                    tinylpu_read_row(
                        dev,
                        TINYLPU_MEM0_OFFSET,
                        MICROGPT_SOFTMAX_OUT_BASE + (uint32_t)pos
                    ),
                    probability_lanes,
                    &probability_scale
                );
                // Each of the eight duplicated softmax lanes holds p/8. An
                // exponent +3 presents p to MXM without ARM multiplication.
                probability = pack_quant_row(
                    probability_lanes,
                    clamp_scale((int)probability_scale + 3)
                );

                int8_t source_lanes[MICROGPT_LANES];
                int8_t value_scale;
                int8_t head_lanes[MICROGPT_LANES] = {0};
                unpack_row(value_rows[pos][row_index], source_lanes, &value_scale);
                for (int lane = lane_base; lane < lane_base + MICROGPT_HEAD_DIM; ++lane) {
                    head_lanes[lane] = source_lanes[lane];
                }
                value = pack_quant_row(head_lanes, value_scale);
            }
            tinylpu_write_row(
                dev,
                TINYLPU_MEM0_OFFSET,
                MICROGPT_MEM0_ATTN_PV_PROB_BASE + (uint32_t)pos,
                probability
            );
            tinylpu_write_row(
                dev,
                TINYLPU_MEM1_OFFSET,
                MICROGPT_MEM1_ATTN_V_STAGE_BASE + (uint32_t)pos,
                value
            );
        }

        run_attention_section(
            dev,
            "PV",
            MICROGPT_ATTN_PV_START,
            MICROGPT_ATTN_PV_INSTRUCTIONS,
            opt
        );
        tinylpu_copy_row(
            dev,
            TINYLPU_MEM0_OFFSET,
            MICROGPT_MEM0_ATTN_PV_OUT_ROW,
            TINYLPU_MEM0_OFFSET,
            MICROGPT_MEM0_ATTN_HEAD_OUT_BASE + (uint32_t)head
        );
    }

    run_attention_section(
        dev,
        "head merge",
        MICROGPT_ATTN_MERGE_START,
        MICROGPT_ATTN_MERGE_INSTRUCTIONS,
        opt
    );
}

static void decode_logits(tinylpu_mmio_t *dev, double logits[MICROGPT_VOCAB_SIZE], bool verbose) {
    if (verbose) {
        fprintf(stderr, "[logits] read\n");
        fflush(stderr);
    }
    tinylpu_mmio_row_t rows[4] = {
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_LOGIT_ROW0),
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_LOGIT_ROW1),
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_LOGIT_ROW2),
        tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, MICROGPT_MEM0_LOGIT_ROW3),
    };
    if (verbose) {
        for (int r = 0; r < 4; ++r) {
            fprintf(stderr, "[logits] raw row%d=%08x %08x %08x\n",
                    r,
                    rows[r].w0,
                    rows[r].w1,
                    rows[r].w2);
        }
        fflush(stderr);
    }
    int out = 0;
    for (int r = 0; r < 4 && out < MICROGPT_VOCAB_SIZE; ++r) {
        int8_t lanes[MICROGPT_LANES];
        int8_t scale;
        unpack_row(rows[r], lanes, &scale);
        for (int lane = 0; lane < MICROGPT_LANES && out < MICROGPT_VOCAB_SIZE; ++lane) {
            logits[out++] = ldexp((double)lanes[lane], scale);
        }
    }
    if (verbose) {
        int first = 0;
        int second = 0;
        int third = 0;
        for (int i = 1; i < MICROGPT_VOCAB_SIZE; ++i) {
            if (logits[i] > logits[first]) {
                third = second;
                second = first;
                first = i;
            } else if (i != first && (second == first || logits[i] > logits[second])) {
                third = second;
                second = i;
            } else if (i != first && i != second && (third == first || logits[i] > logits[third])) {
                third = i;
            }
        }
        fprintf(stderr,
                "[logits] top: %d('%c')=%g  %d('%c')=%g  %d('%c')=%g\n",
                first,
                token_char(first) ? token_char(first) : '#',
                logits[first],
                second,
                token_char(second) ? token_char(second) : '#',
                logits[second],
                third,
                token_char(third) ? token_char(third) : '#',
                logits[third]);
        fflush(stderr);
    }
}

static void run_token(tinylpu_mmio_t *dev, int token_id, int pos_id, const runtime_options_t *opt, double logits[MICROGPT_VOCAB_SIZE]) {
    if (pos_id >= MICROGPT_BLOCK_SIZE) {
        pos_id = MICROGPT_BLOCK_SIZE - 1;
    }

    write_step_inputs(dev, token_id, pos_id, opt->verbose);
    if (!opt->host_broadcasts) {
        // Execute the compiler's SXM broadcasts in the FPGA.  The only split
        // is where the scheduler/runtime supplies the causal attention row.
        run_program(dev, "prefix-fpga-sxm", 0, MICROGPT_PREFIX_INSTRUCTIONS, opt);
        if (opt->verbose) {
            debug_dump_runtime_rows(dev, "after prefix");
        }
        cache_current_kv(dev, pos_id, opt->verbose);
        if (opt->attention_mode == ATTENTION_HOST) {
            stage_host_attention(dev, pos_id, opt, false);
        } else if (opt->attention_mode == ATTENTION_FPGA_SOFTMAX) {
            stage_host_attention(dev, pos_id, opt, true);
        } else if (opt->attention_mode == ATTENTION_FPGA_MXM) {
            stage_mxm_attention(dev, pos_id, opt);
        } else {
            stage_current_attention(dev, opt->verbose);
        }
        run_program(
            dev,
            "suffix-fpga-sxm",
            MICROGPT_PREFIX_INSTRUCTIONS,
            MICROGPT_IMEM_INSTRUCTIONS - MICROGPT_PREFIX_INSTRUCTIONS,
            opt
        );
        if (opt->verbose) {
            debug_dump_runtime_rows(dev, "after suffix");
        }
        decode_logits(dev, logits, opt->verbose);
        return;
    }

    run_program(dev, "prefix-pre-bcast", 0, MICROGPT_PREFIX_ATTN_BCAST_START, opt);
    if (opt->verbose) {
        fprintf(stderr, "[broadcast] stage attention input XN -> MXM rows\n");
        fflush(stderr);
    }
    stage_broadcast_pair(dev, MICROGPT_MEM0_XN_ROW0, MICROGPT_MEM0_XN_ROW1, MICROGPT_MEM0_X_BCAST_BASE);
    run_program(
        dev,
        "prefix-qkv",
        MICROGPT_PREFIX_WQ_START,
        MICROGPT_PREFIX_INSTRUCTIONS - MICROGPT_PREFIX_WQ_START,
        opt
    );
    if (opt->verbose) {
        debug_dump_runtime_rows(dev, "after prefix");
    }
    cache_current_kv(dev, pos_id, opt->verbose);
    if (opt->attention_mode == ATTENTION_HOST) {
        stage_host_attention(dev, pos_id, opt, false);
    } else if (opt->attention_mode == ATTENTION_FPGA_SOFTMAX) {
        stage_host_attention(dev, pos_id, opt, true);
    } else if (opt->attention_mode == ATTENTION_FPGA_MXM) {
        stage_mxm_attention(dev, pos_id, opt);
    } else {
        stage_current_attention(dev, opt->verbose);
    }
    if (opt->verbose) {
        debug_dump_runtime_rows(dev, "after attention");
    }
    if (opt->verbose) {
        fprintf(stderr, "[broadcast] stage attention output -> MXM rows\n");
        fflush(stderr);
    }
    stage_broadcast_pair(dev, MICROGPT_MEM0_ATTN_ROW0, MICROGPT_MEM0_ATTN_ROW1, MICROGPT_MEM0_ATTN_BCAST_BASE);
    run_program(
        dev,
        "suffix-attn",
        MICROGPT_SUFFIX_ATTN_PROJ_START,
        MICROGPT_SUFFIX_MLP_BCAST_START - MICROGPT_SUFFIX_ATTN_PROJ_START,
        opt
    );
    if (opt->verbose) {
        fprintf(stderr, "[broadcast] stage MLP input XN -> MXM rows\n");
        fflush(stderr);
    }
    stage_broadcast_pair(dev, MICROGPT_MEM0_XN_ROW0, MICROGPT_MEM0_XN_ROW1, MICROGPT_MEM0_MLP_BCAST_BASE);
    run_program(
        dev,
        "suffix-mlp-fc1",
        MICROGPT_SUFFIX_MLP_FC1_START,
        g_microgpt_hidden_bcast_start[0] - MICROGPT_SUFFIX_MLP_FC1_START,
        opt
    );
    for (uint32_t block = 0; block < 8u; ++block) {
        uint32_t current_pc = (block == 0u) ? g_microgpt_hidden_bcast_start[0] : g_microgpt_hidden_bcast_end[block - 1u];
        if (g_microgpt_hidden_bcast_start[block] > current_pc) {
            run_program(
                dev,
                "suffix-mlp-relu",
                current_pc,
                g_microgpt_hidden_bcast_start[block] - current_pc,
                opt
            );
        }
        if (opt->verbose) {
            fprintf(stderr, "[broadcast] stage MLP hidden block %u -> MXM rows\n", block);
            fflush(stderr);
        }
        stage_broadcast_row(
            dev,
            MICROGPT_MEM0_MLP_H_BASE + block,
            MICROGPT_MEM0_MLP_H_BCAST_BASE + block * MICROGPT_LANES
        );
    }
    run_program(
        dev,
        "suffix-tail-pre-final-bcast",
        g_microgpt_hidden_bcast_end[7],
        MICROGPT_SUFFIX_FINAL_BCAST_START - g_microgpt_hidden_bcast_end[7],
        opt
    );
    if (opt->verbose) {
        fprintf(stderr, "[broadcast] stage final residual -> LM-head rows\n");
        fflush(stderr);
    }
    stage_broadcast_pair(dev, MICROGPT_MEM0_X_ROW0, MICROGPT_MEM0_X_ROW1, MICROGPT_MEM0_X_BCAST_BASE);
    run_program(
        dev,
        "suffix-lm-head",
        MICROGPT_SUFFIX_LM_HEAD_START,
        MICROGPT_IMEM_INSTRUCTIONS - MICROGPT_SUFFIX_LM_HEAD_START,
        opt
    );
    if (opt->verbose) {
        debug_dump_runtime_rows(dev, "after suffix");
    }
    decode_logits(dev, logits, opt->verbose);
}

static int encode_prompt(const char *prompt, int tokens[MICROGPT_BLOCK_SIZE]) {
    const char *chars = MICROGPT_TOKEN_CHARS;
    int count = 0;
    tokens[count++] = MICROGPT_BOS_TOKEN_ID;
    for (const char *p = prompt; *p && count < MICROGPT_BLOCK_SIZE; ++p) {
        char ch = (char)tolower((unsigned char)*p);
        const char *found = strchr(chars, ch);
        if (found) {
            tokens[count++] = (int)(found - chars);
        }
    }
    return count;
}

static int greedy_next(const double logits[MICROGPT_VOCAB_SIZE]) {
    int best = 0;
    double best_v = logits[0];
    for (int i = 1; i < MICROGPT_VOCAB_SIZE; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

static char token_char(int token_id) {
    const char *chars = MICROGPT_TOKEN_CHARS;
    if (token_id >= 0 && token_id < (int)strlen(chars)) {
        return chars[token_id];
    }
    return '\0';
}

static bool is_target_name(const char *text) {
    for (size_t i = 0; i < MICROGPT_TARGET_NAME_COUNT; ++i) {
        if (strcmp(text, g_microgpt_target_names[i]) == 0) {
            return true;
        }
    }
    return false;
}

static bool target_has_prefix(const char *text) {
    size_t prefix_len = strlen(text);
    for (size_t i = 0; i < MICROGPT_TARGET_NAME_COUNT; ++i) {
        if (strncmp(g_microgpt_target_names[i], text, prefix_len) == 0) {
            return true;
        }
    }
    return false;
}

static int constrained_next(const double logits[MICROGPT_VOCAB_SIZE], const char *emitted) {
    bool allowed[MICROGPT_VOCAB_SIZE] = {false};
    bool have_allowed = false;
    size_t emitted_len = strlen(emitted);

    for (size_t i = 0; i < MICROGPT_TARGET_NAME_COUNT; ++i) {
        const char *target = g_microgpt_target_names[i];
        if (strncmp(target, emitted, emitted_len) == 0 && target[emitted_len] != '\0') {
            const char *found = strchr(MICROGPT_TOKEN_CHARS, target[emitted_len]);
            if (found) {
                int token_id = (int)(found - MICROGPT_TOKEN_CHARS);
                if (token_id >= 0 && token_id < MICROGPT_VOCAB_SIZE) {
                    allowed[token_id] = true;
                    have_allowed = true;
                }
            }
        }
    }

    if (!have_allowed) {
        return greedy_next(logits);
    }

    int best = -1;
    double best_v = 0.0;
    for (int i = 0; i < MICROGPT_VOCAB_SIZE; ++i) {
        if (!allowed[i]) {
            continue;
        }
        if (best < 0 || logits[i] > best_v) {
            best = i;
            best_v = logits[i];
        }
    }
    return best >= 0 ? best : greedy_next(logits);
}

static void generate(tinylpu_mmio_t *dev, const char *prompt, const runtime_options_t *opt) {
    const double request_start = monotonic_seconds();
    reset_prompt_state(dev, opt);

    int tokens[MICROGPT_BLOCK_SIZE];
    int count = encode_prompt(prompt, tokens);
    double logits[MICROGPT_VOCAB_SIZE] = {0};
    char emitted[256];
    size_t emitted_len = 0;
    unsigned output_tokens = 0;
    unsigned decode_steps = 0;
    double decode_step_seconds = 0.0;
    double first_token_time = 0.0;

    const double prefill_start = monotonic_seconds();
    for (int pos = 0; pos < count; ++pos) {
        if (opt->verbose) {
            fprintf(stderr, "[prefill] pos=%d/%d token=%d\n", pos + 1, count, tokens[pos]);
            fflush(stderr);
        }
        run_token(dev, tokens[pos], pos, opt, logits);
    }
    const double prefill_end = monotonic_seconds();

    for (int i = 1; i < count; ++i) {
        char ch = token_char(tokens[i]);
        if (ch) {
            putchar(ch);
            if (emitted_len + 1u < sizeof(emitted)) {
                emitted[emitted_len++] = ch;
                emitted[emitted_len] = '\0';
            }
        }
    }
    fflush(stdout);

    if (emitted_len > 0u && is_target_name(emitted)) {
        putchar('\n');
        if (opt->benchmark) {
            const double request_end = monotonic_seconds();
            print_benchmark(
                count - 1,
                0,
                (unsigned)count,
                0,
                request_end - request_start,
                prefill_end - prefill_start,
                0.0,
                0.0);
        }
        return;
    }

    for (unsigned step = 0; step < opt->max_new_tokens; ++step) {
        int next = opt->decode_mode == DECODE_TARGET && target_has_prefix(emitted)
            ? constrained_next(logits, emitted)
            : greedy_next(logits);
        if (next == MICROGPT_BOS_TOKEN_ID) {
            break;
        }
        char ch = token_char(next);
        if (!ch) {
            break;
        }
        putchar(ch);
        ++output_tokens;
        if (first_token_time == 0.0) {
            first_token_time = monotonic_seconds();
        }
        if (emitted_len + 1u < sizeof(emitted)) {
            emitted[emitted_len++] = ch;
            emitted[emitted_len] = '\0';
        }
        fflush(stdout);

        if (is_target_name(emitted)) {
            break;
        }

        int pos = count + (int)step;
        if (opt->verbose) {
            fprintf(stderr, "\n[generate] step=%u token=%d pos=%d\n", step + 1, next, pos);
            fflush(stderr);
        }
        const double decode_step_start = monotonic_seconds();
        run_token(dev, next, pos, opt, logits);
        decode_step_seconds += monotonic_seconds() - decode_step_start;
        ++decode_steps;
    }
    putchar('\n');
    if (opt->benchmark) {
        const double request_end = monotonic_seconds();
        print_benchmark(
            count - 1,
            output_tokens,
            (unsigned)count,
            decode_steps,
            request_end - request_start,
            prefill_end - prefill_start,
            decode_step_seconds,
            first_token_time > 0.0 ? first_token_time - request_start : 0.0);
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --base HEX              physical lightweight bridge base (default 0x%08X)\n"
        "  --span HEX              mmap span (default 0x%08X)\n"
        "  --settle-us N           poll delay while exact-cycle page run is active (default 1000)\n"
        "  --max-new-tokens N      generated chars per prompt (default 12)\n"
        "  --no-load-weights       assume MEM1 already contains model rows\n"
        "  --verbose               print progress for each token/page/MMIO stage\n"
        "  --benchmark             report full ARM+FPGA wall-clock throughput per prompt\n"
        "  --broadcast sxm         execute compiled broadcasts on FPGA SXM (diagnostic)\n"
        "  --broadcast host        ARM stages broadcast rows (board-safe default)\n"
        "  --probe-only            map bridge, write/read a few control regs, then exit\n"
        "  --sxm-probe             run one known-pattern FPGA SXM broadcast and exit\n"
        "  --attention host        ARM computes tiny causal attention context from FPGA K/V cache\n"
        "  --attention fpga-mxm    FPGA SXM K^T, MXM QK/PV, and VXM softmax (default)\n"
        "  --attention fpga-softmax FPGA softmax; ARM computes QK/PV (diagnostic)\n"
        "  --attention current     no ARM attention math; stage current V as context\n"
        "  --decode greedy         raw model argmax every step (default)\n"
        "  --decode target         constrain generated chars to exported target names\n",
        argv0,
        TINYLPU_HPS_LW_BRIDGE_BASE_DEFAULT,
        TINYLPU_HPS_LW_BRIDGE_SPAN_DEFAULT);
}

static int parse_u32(const char *text, uint32_t *out) {
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 0);
    if (errno || !end || *end != '\0' || value > 0xFFFFFFFFul) {
        return -1;
    }
    *out = (uint32_t)value;
    return 0;
}

static int parse_args(int argc, char **argv, runtime_options_t *opt) {
    opt->base = TINYLPU_HPS_LW_BRIDGE_BASE_DEFAULT;
    opt->span = TINYLPU_HPS_LW_BRIDGE_SPAN_DEFAULT;
    opt->settle_us = 1000;
    opt->skip_load_weights = false;
    opt->verbose = false;
    opt->probe_only = false;
    opt->sxm_probe = false;
    opt->benchmark = false;
    opt->host_broadcasts = true;
    opt->attention_mode = ATTENTION_FPGA_MXM;
    opt->decode_mode = DECODE_GREEDY;
    opt->max_new_tokens = 12;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) {
            uint32_t value;
            if (parse_u32(argv[++i], &value) != 0) return -1;
            opt->base = value;
        } else if (strcmp(argv[i], "--span") == 0 && i + 1 < argc) {
            uint32_t value;
            if (parse_u32(argv[++i], &value) != 0) return -1;
            opt->span = value;
        } else if (strcmp(argv[i], "--settle-us") == 0 && i + 1 < argc) {
            uint32_t value;
            if (parse_u32(argv[++i], &value) != 0) return -1;
            opt->settle_us = value;
        } else if (strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            uint32_t value;
            if (parse_u32(argv[++i], &value) != 0) return -1;
            opt->max_new_tokens = value;
        } else if (strcmp(argv[i], "--no-load-weights") == 0) {
            opt->skip_load_weights = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            opt->verbose = true;
        } else if (strcmp(argv[i], "--probe-only") == 0) {
            opt->probe_only = true;
        } else if (strcmp(argv[i], "--sxm-probe") == 0) {
            opt->sxm_probe = true;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            opt->benchmark = true;
        } else if (strcmp(argv[i], "--broadcast") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "sxm") == 0) {
                opt->host_broadcasts = false;
            } else if (strcmp(mode, "host") == 0) {
                opt->host_broadcasts = true;
            } else {
                return -1;
            }
        } else if (strcmp(argv[i], "--attention") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "host") == 0) {
                opt->attention_mode = ATTENTION_HOST;
            } else if (strcmp(mode, "fpga-softmax") == 0) {
                opt->attention_mode = ATTENTION_FPGA_SOFTMAX;
            } else if (strcmp(mode, "fpga-mxm") == 0) {
                opt->attention_mode = ATTENTION_FPGA_MXM;
            } else if (strcmp(mode, "current") == 0) {
                opt->attention_mode = ATTENTION_CURRENT;
            } else {
                return -1;
            }
        } else if (strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "target") == 0) {
                opt->decode_mode = DECODE_TARGET;
            } else if (strcmp(mode, "greedy") == 0) {
                opt->decode_mode = DECODE_GREEDY;
            } else {
                return -1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return -1;
        }
    }
    return 0;
}

static int probe_bridge(tinylpu_mmio_t *dev) {
    puts("Probe: HPS lightweight bridge control registers");
    tinylpu_write32(dev, TINYLPU_CTRL_RUN, 0);
    tinylpu_write32(dev, TINYLPU_CTRL_PC_LOAD, 0);
    uint32_t run = tinylpu_read32(dev, TINYLPU_CTRL_RUN);
    uint32_t cycles0 = tinylpu_read32(dev, TINYLPU_CTRL_CYCLES);
    usleep(1000);
    uint32_t cycles1 = tinylpu_read32(dev, TINYLPU_CTRL_CYCLES);
    printf("Probe: CTRL_RUN=0x%08x CYCLES=%u->%u\n", run, cycles0, cycles1);

    puts("Probe: soft reset control");
    tinylpu_soft_reset(dev, 32u, 10u);
    uint32_t reset_after = tinylpu_read32(dev, TINYLPU_CTRL_SOFT_RESET);
    printf("Probe: SOFT_RESET remaining=%u\n", reset_after);
    if (reset_after != 0u) {
        fprintf(stderr,
                "Probe failed: soft reset register is not behaving. Recompile/reprogram the latest HPS .sof.\n");
        return 1;
    }

    puts("Probe: exact-cycle run control");
    uint32_t exact_cycles = tinylpu_run_cycles(dev, 32u, 10u);
    uint32_t run_after = tinylpu_read32(dev, TINYLPU_CTRL_RUN);
    uint32_t remaining_after = tinylpu_read32(dev, TINYLPU_CTRL_RUN_CYCLES);
    uint32_t cycles2 = tinylpu_read32(dev, TINYLPU_CTRL_CYCLES);
    printf("Probe: RUN_CYCLES requested=32 actual=%u run=%u remaining=%u CYCLES=%u\n",
           exact_cycles,
           run_after,
           remaining_after,
           cycles2);
    if (exact_cycles == 0u || run_after != 0u || remaining_after != 0u) {
        fprintf(stderr,
                "Probe failed: exact-cycle register is not behaving. Reprogram the latest HPS .sof before inference.\n");
        return 1;
    }

    puts("Probe: MEM0 row write/read");
    tinylpu_mmio_row_t pattern = {0x11223344u, 0x55667788u, 0x99aabbccu};
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, 15, pattern);
    tinylpu_mmio_row_t got = tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, 15);
    printf("Probe: MEM0[15]=%08x %08x %08x\n", got.w0, got.w1, got.w2);

    puts("Probe: MEM1 row write/read");
    tinylpu_write_row(dev, TINYLPU_MEM1_OFFSET, 15, pattern);
    got = tinylpu_read_row(dev, TINYLPU_MEM1_OFFSET, 15);
    printf("Probe: MEM1[15]=%08x %08x %08x\n", got.w0, got.w1, got.w2);
    fflush(stdout);
    return 0;
}

static int probe_sxm(tinylpu_mmio_t *dev, const runtime_options_t *opt) {
    static const int8_t source[MICROGPT_LANES] = {
        -91, -37, -5, 0, 7, 29, 63, 111
    };
    const int8_t source_scale = -6;
    const size_t one_broadcast_instructions =
        (MICROGPT_PREFIX_WQ_START - MICROGPT_PREFIX_ATTN_BCAST_START) / 2u;
    tinylpu_mmio_row_t zero = {0, 0, 0};

    puts("SXM probe: source lanes = [-91 -37 -5 0 7 29 63 111], scale=-6");
    tinylpu_soft_reset(dev, 32u, opt->settle_us);
    tinylpu_write_row(
        dev,
        TINYLPU_MEM0_OFFSET,
        MICROGPT_MEM0_XN_ROW0,
        pack_quant_row(source, source_scale)
    );
    for (uint32_t row = 0; row < MICROGPT_LANES; ++row) {
        tinylpu_write_row(
            dev,
            TINYLPU_MEM0_OFFSET,
            MICROGPT_MEM0_X_BCAST_BASE + row,
            zero
        );
    }

    run_program(
        dev,
        "sxm-probe",
        MICROGPT_PREFIX_ATTN_BCAST_START,
        one_broadcast_instructions,
        opt
    );

    int failures = 0;
    for (int row = 0; row < MICROGPT_LANES; ++row) {
        int8_t lanes[MICROGPT_LANES];
        int8_t scale;
        unpack_row(
            tinylpu_read_row(
                dev,
                TINYLPU_MEM0_OFFSET,
                MICROGPT_MEM0_X_BCAST_BASE + (uint32_t)row
            ),
            lanes,
            &scale
        );
        printf("SXM probe row %d: [%d %d %d %d %d %d %d %d] scale=%d%s\n",
               row,
               lanes[0], lanes[1], lanes[2], lanes[3],
               lanes[4], lanes[5], lanes[6], lanes[7],
               scale,
               (scale == source_scale &&
                lanes[0] == source[row] && lanes[1] == source[row] &&
                lanes[2] == source[row] && lanes[3] == source[row] &&
                lanes[4] == source[row] && lanes[5] == source[row] &&
                lanes[6] == source[row] && lanes[7] == source[row]) ? " PASS" : " FAIL");
        if (scale != source_scale) {
            ++failures;
            continue;
        }
        for (int lane = 0; lane < MICROGPT_LANES; ++lane) {
            if (lanes[lane] != source[row]) {
                ++failures;
                break;
            }
        }
    }
    printf("SXM probe: %s (%d bad rows)\n", failures ? "FAIL" : "PASS", failures);
    fflush(stdout);
    tinylpu_soft_reset(dev, 32u, opt->settle_us);
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    runtime_options_t opt;
    if (parse_args(argc, argv, &opt) != 0) {
        usage(argv[0]);
        return 2;
    }

    tinylpu_mmio_t dev;
    if (tinylpu_mmio_open(&dev, opt.base, opt.span) != 0) {
        return 1;
    }

    printf("TinyLPU MicroGPT HPS/Linux runtime\n");
    printf("MMIO base: 0x%08lx, span: 0x%zx\n", (unsigned long)opt.base, opt.span);
    printf("VLIW: %u instructions (%u prefix, %u suffix), MEM1: %u rows\n",
           MICROGPT_IMEM_INSTRUCTIONS,
           MICROGPT_PREFIX_INSTRUCTIONS,
           MICROGPT_SUFFIX_INSTRUCTIONS,
           MICROGPT_MEM1_ROWS);
    const char *attention_name = opt.attention_mode == ATTENTION_HOST
        ? "ARM causal attention"
        : (opt.attention_mode == ATTENTION_FPGA_SOFTMAX
            ? "FPGA softmax with ARM QK/PV staging"
            : (opt.attention_mode == ATTENTION_FPGA_MXM
                ? "FPGA SXM K^T + MXM QK/PV + VXM softmax"
                : "current-token FPGA V only"));
    printf("attention: %s\n", attention_name);
    printf("decode: %s\n", opt.decode_mode == DECODE_TARGET ? "target-name constrained" : "raw greedy");
    printf("broadcast: %s\n", opt.host_broadcasts ? "ARM compatibility staging" : "FPGA SXM");

    if (opt.probe_only) {
        int rc = probe_bridge(&dev);
        tinylpu_mmio_close(&dev);
        return rc;
    }

    if (opt.sxm_probe) {
        int rc = probe_sxm(&dev, &opt);
        tinylpu_mmio_close(&dev);
        return rc;
    }

    if (!opt.skip_load_weights) {
        printf("Loading MEM1/model rows over HPS bridge...\n");
        fflush(stdout);
        const double load_start = monotonic_seconds();
        load_mem1(&dev, opt.verbose);
        printf("MEM1 load complete.\n");
        if (opt.benchmark) {
            fprintf(stderr, "[perf] model load over ARM/MMIO: %.3f s (%u rows)\n",
                    monotonic_seconds() - load_start, MICROGPT_MEM1_ROWS);
        }
    }

    char prompt[256];
    while (true) {
        printf("Prompt > ");
        fflush(stdout);
        if (!fgets(prompt, sizeof(prompt), stdin)) {
            break;
        }
        prompt[strcspn(prompt, "\r\n")] = '\0';
        if (strcmp(prompt, "exit") == 0 || strcmp(prompt, "quit") == 0) {
            break;
        }
        if (prompt[0] == '\0') {
            continue;
        }
        generate(&dev, prompt, &opt);
    }

    tinylpu_mmio_close(&dev);
    return 0;
}
