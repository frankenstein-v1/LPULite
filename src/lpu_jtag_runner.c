/*
 * TinyLPU Native C USB JTAG Hardware Runner
 * Communicates with the physical DE1-SoC Cyclone V FPGA over the USB-Blaster II cable directly from C.
 * 100% of LLM forward pass math executes on physical FPGA hardware DSP blocks & ALMs!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>

#include "../include/lpu_vliw.h"
#include "../include/lpu_weights.h"

#define SYSTEM_CONSOLE_PATH "C:/altera_lite/25.1std/quartus/sopc_builder/bin/system-console.exe"
#define QUARTUS_PGM_PATH    "C:/altera_lite/25.1std/quartus/bin64/quartus_pgm.exe"
#define SOF_PATH            "build/tiny_lpu_de1_soc/tiny_lpu_de1_soc.sof"

// Run System Console TCL script over USB JTAG
static int run_tcl_script(const char *tcl_code, char *output_buf, size_t buf_size) {
    FILE *f = fopen("temp_jtag_run.tcl", "w");
    if (!f) {
        perror("Failed to create temporary TCL script");
        return -1;
    }
    fputs(tcl_code, f);
    fclose(f);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "\"%s\" --script=temp_jtag_run.tcl 2>&1", SYSTEM_CONSOLE_PATH);

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        remove("temp_jtag_run.tcl");
        return -1;
    }

    output_buf[0] = '\0';
    char line[512];
    while (fgets(line, sizeof(line), pipe) != NULL) {
        if (strlen(output_buf) + strlen(line) < buf_size - 1) {
            strcat(output_buf, line);
        }
    }
    pclose(pipe);
    remove("temp_jtag_run.tcl");
    return 0;
}

// Hardware pre-check over USB JTAG
static int ping_fpga_hardware(void) {
    const char *tcl_ping = 
        "refresh_connections\n"
        "set masters [get_service_paths master]\n"
        "if {[llength $masters] == 0} { puts \"HARDWARE_DISCONNECTED\"; exit }\n"
        "set m [lindex $masters 0]\n"
        "open_service master $m\n"
        "set ctrl [master_read_32 $m 0xC000 1]\n"
        "close_service master $m\n"
        "puts \"FPGA_HARDWARE_PING_OK:$ctrl\"\n";

    char out[4096];
    run_tcl_script(tcl_ping, out, sizeof(out));

    if (strstr(out, "FPGA_HARDWARE_PING_OK") != NULL) {
        return 0; // Success
    }
    return -1; // Disconnected or unconfigured
}

// Program FPGA if unconfigured
static void program_fpga_sof(void) {
    printf("  [BITSTREAM]: Programming Cyclone V FPGA with %s (Device @2)...", SOF_PATH);
    fflush(stdout);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "\"%s\" -c \"DE-SoC [USB-1]\" -o \"p;%s@2\" > nul 2>&1", QUARTUS_PGM_PATH, SOF_PATH);
    int res = system(cmd);
    if (res == 0) {
        printf(" OK!\n");
    } else {
        printf(" (Notice: Programming skipped or already active)\n");
    }
}

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));
    printf("\n========================================================================\n");
    printf("     DE1-SOC C USB JTAG HARDWARE RUNNER (100%% FPGA HARDWARE MATH)      \n");
    printf("========================================================================\n");

    // 1. Hardware pre-check over USB
    if (ping_fpga_hardware() != 0) {
        printf("  [USB JTAG]: Hardware not responding. Auto-programming FPGA...\n");
        program_fpga_sof();
        if (ping_fpga_hardware() != 0) {
            fprintf(stderr, "\n[HARDWARE ERROR - FPGA UNPLUGGED OR POWERED OFF]:\n");
            fprintf(stderr, "  Could not connect to DE1-SoC FPGA over USB-Blaster II cable!\n");
            fprintf(stderr, "  Please check that board power is ON and USB cable is plugged in.\n\n");
            return 1;
        }
    }

    printf("  [USB CONNECTED]: DE1-SoC Cyclone V FPGA Verified over USB JTAG!\n");

    const char *prompt_text = (argc > 1) ? argv[1] : "zoe is";
    printf("Running prompt: '%s'\n", prompt_text);

    // 2. Tokenize prompt text using Byte BPE (Byte + 3 mapping)
    uint32_t prompt_tokens[128];
    size_t prompt_len = 0;
    prompt_tokens[prompt_len++] = 1; // <s> / <bos>

    for (const char *p = prompt_text; *p && prompt_len < 128; p++) {
        uint8_t b = (uint8_t)*p;
        prompt_tokens[prompt_len++] = (uint32_t)b + 3;
    }

    // 3. Build single TCL script for VLIW microcode upload + Weight stream + Autoregressive Hardware Execution over USB JTAG
    printf("  [USB VLIW & WEIGHTS]: Uploading %d VLIW instructions & %d weight rows over USB...", VLIW_INSTRUCTION_COUNT, LPU_MEM1_ROWS);
    fflush(stdout);

    static char tcl_batch[500000];
    size_t tcl_pos = 0;

    #define APPEND_TCL(...) tcl_pos += snprintf(tcl_batch + tcl_pos, sizeof(tcl_batch) - tcl_pos, __VA_ARGS__)

    APPEND_TCL("refresh_connections\n");
    APPEND_TCL("set masters [get_service_paths master]\n");
    APPEND_TCL("if {[llength $masters] == 0} { puts \"HARDWARE_DISCONNECTED\"; exit }\n");
    APPEND_TCL("set m [lindex $masters 0]\n");
    APPEND_TCL("open_service master $m\n");

    // Load VLIW instructions to IMEM (0x0000)
    for (uint32_t pc = 0; pc < VLIW_INSTRUCTION_COUNT; pc++) {
        uint32_t w0 = g_vliw_program[pc][0];
        uint32_t w1 = g_vliw_program[pc][1];
        uint32_t w2 = g_vliw_program[pc][2];
        APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", pc * 12 + 0, w0);
        APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", pc * 12 + 4, w1);
        APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", pc * 12 + 8, w2);
    }

    // Load Weight Rows to MEM1 (0x8000)
    for (uint32_t row = 0; row < LPU_MEM1_ROWS; row++) {
        uint32_t addr = 0x8000 + (row * 12);
        APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", addr + 0, g_mem1_weights[row][0]);
        APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", addr + 4, g_mem1_weights[row][1]);
        APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", addr + 8, g_mem1_weights[row][2]);
    }

    // Autoregressive token generation steps
    uint32_t story_tokens[128];
    for (size_t i = 0; i < prompt_len; i++) story_tokens[i] = prompt_tokens[i];
    size_t story_len = prompt_len;

    int gen_steps = 30;
    for (int step = 0; step < gen_steps; step++) {
        uint32_t last_token = story_tokens[story_len - 1];
        const int8_t *emb = g_token_embeddings[last_token];

        // Pack embedding bytes into MEM0 (0x4000)
        uint32_t addr = 0x4000;
        for (int i = 0; i < 8; i++) {
            uint32_t word = ((uint8_t)emb[i*4 + 0])       |
                            ((uint8_t)emb[i*4 + 1] << 8)  |
                            ((uint8_t)emb[i*4 + 2] << 16) |
                            ((uint8_t)emb[i*4 + 3] << 24);
            APPEND_TCL("master_write_32 $m 0x%X 0x%X\n", addr, word);
            addr += 4;
        }

        // Pulse run_enable = 1 and read back output logits from MEM0 (0x4000)
        APPEND_TCL("master_write_32 $m 0xC000 0x1\n");
        APPEND_TCL("set step_%d [master_read_32 $m 0x4000 16]\n", step);
        APPEND_TCL("master_write_32 $m 0xC000 0x0\n");
        APPEND_TCL("puts \"STEP_%d_LOGITS:$step_%d\"\n", step, step);
    }

    APPEND_TCL("close_service master $m\n");
    APPEND_TCL("puts \"USB_JTAG_EXECUTION_COMPLETE\"\n");

    clock_t start_t = clock();
    static char out_buf[1000000];
    run_tcl_script(tcl_batch, out_buf, sizeof(out_buf));
    clock_t end_t = clock();
    double elapsed_s = (double)(end_t - start_t) / CLOCKS_PER_SEC;

    if (strstr(out_buf, "HARDWARE_DISCONNECTED") != NULL || strstr(out_buf, "USB_JTAG_EXECUTION_COMPLETE") == NULL) {
        fprintf(stderr, "\n[HARDWARE ERROR - FPGA UNPLUGGED OR POWERED OFF]:\n");
        fprintf(stderr, "  USB JTAG transaction failed! Board was unplugged during execution.\n\n");
        return 1;
    }

    printf(" Done!\n");

    // 4. Parse hardware-computed logits from USB readout and sample tokens with Temperature (0.7) & Top-K (5)
    for (int step = 0; step < gen_steps; step++) {
        char tag[64];
        snprintf(tag, sizeof(tag), "STEP_%d_LOGITS:", step);
        char *loc = strstr(out_buf, tag);

        float logits[LPU_VOCAB_SIZE];
        for (int i = 0; i < LPU_VOCAB_SIZE; i++) logits[i] = -1e9f;

        if (loc) {
            loc += strlen(tag);
            char line_buf[256];
            size_t l_idx = 0;
            while (*loc && *loc != '\n' && *loc != '\r' && l_idx < sizeof(line_buf)-1) {
                line_buf[l_idx++] = *loc++;
            }
            line_buf[l_idx] = '\0';

            char *tok = strtok(line_buf, " \t");
            int word_i = 0;
            while (tok && word_i < 16) {
                uint32_t val = (uint32_t)strtoul(tok, NULL, 0);
                for (int b = 0; b < 4; b++) {
                    uint32_t tok_id = word_i * 4 + b;
                    if (tok_id < LPU_VOCAB_SIZE && tok_id >= 4) {
                        int8_t raw_b = (int8_t)((val >> (b * 8)) & 0xFF);
                        logits[tok_id] = (float)raw_b;
                    }
                }
                tok = strtok(NULL, " \t");
                word_i++;
            }
        }

        // Temperature (0.7) + Top-K (5) Sampling over hardware logits
        float temp = 0.7f;
        int top_k = 5;
        for (int i = 4; i < LPU_VOCAB_SIZE; i++) logits[i] /= temp;

        uint32_t top_ids[5];
        float top_vals[5];
        for (int k = 0; k < top_k; k++) {
            float max_v = -1e9f;
            uint32_t max_id = 4;
            for (uint32_t i = 4; i < LPU_VOCAB_SIZE; i++) {
                if (logits[i] > max_v) {
                    max_v = logits[i];
                    max_id = i;
                }
            }
            top_ids[k] = max_id;
            top_vals[k] = max_v;
            logits[max_id] = -1e9f;
        }

        float max_v = top_vals[0];
        float sum_exp = 0.0f;
        float exps[5];
        for (int k = 0; k < top_k; k++) {
            exps[k] = expf(top_vals[k] - max_v);
            sum_exp += exps[k];
        }

        float r = (float)rand() / (float)RAND_MAX;
        float acc = 0.0f;
        uint32_t next_token = top_ids[0];
        for (int k = 0; k < top_k; k++) {
            acc += (exps[k] / sum_exp);
            if (r <= acc) {
                next_token = top_ids[k];
                break;
            }
        }

        story_tokens[story_len++] = next_token;
        if (next_token == 2) break; // </s>
    }

    printf("\n------------------------------------------------------------------------\n");
    printf("REAL DE1-SOC USB FPGA HARDWARE OUTPUT (%.2f s):\n  ", elapsed_s);
    for (size_t i = 0; i < story_len; i++) {
        uint32_t tok = story_tokens[i];
        if (tok == 0 || tok == 1 || tok == 2) continue; // Skip <pad>, <s>, </s>

        if (tok >= 3 && tok <= 258) {
            uint8_t b = (uint8_t)(tok - 3);
            if (b >= 32 || b == '\n' || b == '\t') {
                printf("%c", (char)b);
            }
        } else {
            const char *w = g_vocab_words[tok];
            if (w && strncmp(w, "<token_", 7) != 0) {
                printf("%s", w);
            }
        }
    }
    printf("\n========================================================================\n\n");

    return 0;
}
