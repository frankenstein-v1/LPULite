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

typedef enum {
    ATTENTION_CURRENT = 0,
    ATTENTION_HOST = 1,
} attention_mode_t;

typedef struct {
    uintptr_t base;
    size_t span;
    unsigned settle_us;
    bool skip_load_weights;
    bool verbose;
    bool probe_only;
    attention_mode_t attention_mode;
    unsigned max_new_tokens;
} runtime_options_t;

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

static void run_program(tinylpu_mmio_t *dev, const char *label, size_t start_pc, size_t instruction_count, const runtime_options_t *opt) {
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
        for (size_t row = 0; row < TINYLPU_IMEM_ROWS; ++row) {
            tinylpu_mmio_row_t value = {0, 0, 0};
            if (row < page_count) {
                value = as_mmio_row(g_microgpt_vliw[start_pc + done + row]);
            }
            tinylpu_write_row(dev, TINYLPU_IMEM_OFFSET, (uint32_t)row, value);
        }
        if (opt->verbose) {
            fprintf(stderr, "[%s] run page %u\n", label, page);
            fflush(stderr);
        }
        tinylpu_run_cycles(dev, (uint32_t)(page_count + 8u), opt->settle_us);
        if (opt->verbose) {
            fprintf(stderr, "[%s] done page %u\n", label, page);
            fflush(stderr);
        }
        done += page_count;
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

static void stage_host_attention(tinylpu_mmio_t *dev, int through_pos, bool verbose) {
    if (verbose) {
        fprintf(stderr, "[attention] host causal through_pos=%d\n", through_pos);
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
        softmax(scores, through_pos + 1, weights);
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
    int out = 0;
    for (int r = 0; r < 4 && out < MICROGPT_VOCAB_SIZE; ++r) {
        int8_t lanes[MICROGPT_LANES];
        int8_t scale;
        unpack_row(rows[r], lanes, &scale);
        for (int lane = 0; lane < MICROGPT_LANES && out < MICROGPT_VOCAB_SIZE; ++lane) {
            logits[out++] = ldexp((double)lanes[lane], scale);
        }
    }
}

static void run_token(tinylpu_mmio_t *dev, int token_id, int pos_id, const runtime_options_t *opt, double logits[MICROGPT_VOCAB_SIZE]) {
    if (pos_id >= MICROGPT_BLOCK_SIZE) {
        pos_id = MICROGPT_BLOCK_SIZE - 1;
    }

    write_step_inputs(dev, token_id, pos_id, opt->verbose);
    run_program(dev, "prefix", 0, MICROGPT_PREFIX_INSTRUCTIONS, opt);
    cache_current_kv(dev, pos_id, opt->verbose);
    if (opt->attention_mode == ATTENTION_HOST) {
        stage_host_attention(dev, pos_id, opt->verbose);
    } else {
        stage_current_attention(dev, opt->verbose);
    }
    run_program(dev, "suffix", MICROGPT_PREFIX_INSTRUCTIONS, MICROGPT_SUFFIX_INSTRUCTIONS, opt);
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

static void generate(tinylpu_mmio_t *dev, const char *prompt, const runtime_options_t *opt) {
    int tokens[MICROGPT_BLOCK_SIZE];
    int count = encode_prompt(prompt, tokens);
    double logits[MICROGPT_VOCAB_SIZE] = {0};

    for (int pos = 0; pos < count; ++pos) {
        if (opt->verbose) {
            fprintf(stderr, "[prefill] pos=%d/%d token=%d\n", pos + 1, count, tokens[pos]);
            fflush(stderr);
        }
        run_token(dev, tokens[pos], pos, opt, logits);
    }

    for (int i = 1; i < count; ++i) {
        char ch = token_char(tokens[i]);
        if (ch) putchar(ch);
    }
    fflush(stdout);

    for (unsigned step = 0; step < opt->max_new_tokens; ++step) {
        int next = greedy_next(logits);
        if (next == MICROGPT_BOS_TOKEN_ID) {
            break;
        }
        char ch = token_char(next);
        if (!ch) {
            break;
        }
        putchar(ch);
        fflush(stdout);

        int pos = count + (int)step;
        if (opt->verbose) {
            fprintf(stderr, "\n[generate] step=%u token=%d pos=%d\n", step + 1, next, pos);
            fflush(stderr);
        }
        run_token(dev, next, pos, opt, logits);
    }
    putchar('\n');
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
        "  --probe-only            map bridge, write/read a few control regs, then exit\n"
        "  --attention host        ARM computes tiny causal attention context from FPGA K/V cache\n"
        "  --attention current     no ARM attention math; stage current V as context\n",
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
    opt->attention_mode = ATTENTION_HOST;
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
        } else if (strcmp(argv[i], "--attention") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "host") == 0) {
                opt->attention_mode = ATTENTION_HOST;
            } else if (strcmp(mode, "current") == 0) {
                opt->attention_mode = ATTENTION_CURRENT;
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

    puts("Probe: MEM0 row write/read");
    tinylpu_mmio_row_t pattern = {0x11223344u, 0x55667788u, 0x99aabbccu};
    tinylpu_write_row(dev, TINYLPU_MEM0_OFFSET, 15, pattern);
    tinylpu_mmio_row_t got = tinylpu_read_row(dev, TINYLPU_MEM0_OFFSET, 15);
    printf("Probe: MEM0[15]=%08x %08x %08x\n", got.w0, got.w1, got.w2);
    fflush(stdout);
    return 0;
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
    printf("attention: %s\n", opt.attention_mode == ATTENTION_HOST ? "host TB causal attention" : "current-token FPGA V only");

    if (opt.probe_only) {
        int rc = probe_bridge(&dev);
        tinylpu_mmio_close(&dev);
        return rc;
    }

    if (!opt.skip_load_weights) {
        printf("Loading MEM1/model rows over HPS bridge...\n");
        fflush(stdout);
        load_mem1(&dev, opt.verbose);
        printf("MEM1 load complete.\n");
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
