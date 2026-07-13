import json
import math
from pathlib import Path

import cocotb
import lpu_tb as lpu
from cocotb.clock import Clock
from cocotb.handle import Force, Release
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time


ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT_DIR / "tiny_lm_weights_export.json"

MODEL_CONFIG = {
    "vocab_size": 256,
    "dim": 4,
    "seq_len": 4,
    "layers": 1,
    "heads": 1,
    "ffn_dim": 16,
}

PROMPT_PREFILL = ["lebron", "is"]
PROMPT_DECODE = ["lebron", "is", "king"]
PROMPT_PROBE = ["ranvijay", "is"]
FP_CTRL = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)


def load_tiny_lm_export():
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        export = json.load(f)

    vocab = export["vocab"]
    id_to_token = {idx: token for token, idx in vocab.items()}
    return export["config"], vocab, id_to_token, export["weights"]


def vec_add(a_vec, b_vec):
    return [lpu.to_f32(a + b) for a, b in zip(a_vec, b_vec)]


def matmul(a_matrix, b_matrix):
    out = []
    for row in a_matrix:
        out_row = []
        for col_idx in range(len(b_matrix[0])):
            acc = 0.0
            for k_idx, value in enumerate(row):
                product = lpu.to_f32(value * b_matrix[k_idx][col_idx])
                acc = lpu.to_f32(acc + product)
            out_row.append(acc)
        out.append(out_row)
    return out


def transpose(matrix):
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


def linear_no_bias(rows, weight):
    return matmul(rows, transpose(weight))


def add_bias(rows, bias):
    return [[lpu.to_f32(value + bias[col_idx]) for col_idx, value in enumerate(row)] for row in rows]


def linear(rows, weight, bias):
    return add_bias(linear_no_bias(rows, weight), bias)


def layernorm_rows(rows, gamma, beta, eps=1e-5):
    out = []
    for row in rows:
        mean = lpu.to_f32(sum(row) / len(row))
        variance = lpu.to_f32(
            sum(lpu.to_f32((value - mean) * (value - mean)) for value in row) / len(row)
        )
        inv_std = lpu.to_f32(1.0 / math.sqrt(variance + eps))
        out.append([
            lpu.to_f32(lpu.to_f32(lpu.to_f32(value - mean) * inv_std) * gamma[idx] + beta[idx])
            for idx, value in enumerate(row)
        ])
    return out


def softmax_rows(scores):
    out = []
    for row in scores:
        row_max = max(row)
        exp_values = [math.exp(value - row_max) for value in row]
        denom = sum(exp_values)
        out.append([lpu.to_f32(value / denom) for value in exp_values])
    return out


def encode_prompt(tokens, vocab):
    return [vocab.get(token.lower(), vocab["<unk>"]) for token in tokens]


def tiny_lm_forward(input_ids, weights):
    token_emb = weights["token_emb.weight"]
    pos_emb = weights["pos_emb.weight"]

    x0 = [
        vec_add(token_emb[token_id], pos_emb[pos_idx])
        for pos_idx, token_id in enumerate(input_ids)
    ]

    ln1 = layernorm_rows(x0, weights["blocks.0.ln1.weight"], weights["blocks.0.ln1.bias"])
    q_no_bias = linear_no_bias(ln1, weights["blocks.0.attn.q_proj.weight"])
    k_no_bias = linear_no_bias(ln1, weights["blocks.0.attn.k_proj.weight"])
    v_no_bias = linear_no_bias(ln1, weights["blocks.0.attn.v_proj.weight"])
    q = add_bias(q_no_bias, weights["blocks.0.attn.q_proj.bias"])
    k = add_bias(k_no_bias, weights["blocks.0.attn.k_proj.bias"])
    v = add_bias(v_no_bias, weights["blocks.0.attn.v_proj.bias"])

    scores_raw = matmul(q, transpose(k))
    scores = []
    scale = 1.0 / math.sqrt(len(q[0]))
    for row_idx, row in enumerate(scores_raw):
        score_row = []
        for col_idx, value in enumerate(row):
            if col_idx > row_idx:
                score_row.append(-1.0e30)
            else:
                score_row.append(lpu.to_f32(value * scale))
        scores.append(score_row)

    probs = softmax_rows(scores)
    attn = matmul(probs, v)
    attn_out_no_bias = linear_no_bias(attn, weights["blocks.0.attn.out_proj.weight"])
    attn_out = add_bias(attn_out_no_bias, weights["blocks.0.attn.out_proj.bias"])
    x_after_attn = [vec_add(row, attn_out[row_idx]) for row_idx, row in enumerate(x0)]

    ln2 = layernorm_rows(
        x_after_attn,
        weights["blocks.0.ln2.weight"],
        weights["blocks.0.ln2.bias"],
    )
    ffn_hidden_no_bias = linear_no_bias(ln2, weights["blocks.0.ffn.0.weight"])
    ffn_hidden_pre_relu = add_bias(ffn_hidden_no_bias, weights["blocks.0.ffn.0.bias"])
    ffn_hidden = [[lpu.to_f32(max(0.0, value)) for value in row] for row in ffn_hidden_pre_relu]
    ffn_out_no_bias = linear_no_bias(ffn_hidden, weights["blocks.0.ffn.2.weight"])
    ffn_out = add_bias(ffn_out_no_bias, weights["blocks.0.ffn.2.bias"])
    x_after_ffn = [vec_add(row, ffn_out[row_idx]) for row_idx, row in enumerate(x_after_attn)]

    final = layernorm_rows(x_after_ffn, weights["ln_f.weight"], weights["ln_f.bias"])
    logits_no_bias = linear_no_bias(final, weights["lm_head.weight"])
    logits = add_bias(logits_no_bias, weights["lm_head.bias"])
    return {
        "x0": x0,
        "ln1": ln1,
        "q_no_bias": q_no_bias,
        "k_no_bias": k_no_bias,
        "v_no_bias": v_no_bias,
        "q": q,
        "k": k,
        "v": v,
        "scores_raw": scores_raw,
        "scores": scores,
        "probs": probs,
        "attn": attn,
        "attn_out_no_bias": attn_out_no_bias,
        "attn_out": attn_out,
        "x_after_attn": x_after_attn,
        "ln2": ln2,
        "ffn_hidden_no_bias": ffn_hidden_no_bias,
        "ffn_hidden_pre_relu": ffn_hidden_pre_relu,
        "ffn_hidden": ffn_hidden,
        "ffn_out_no_bias": ffn_out_no_bias,
        "ffn_out": ffn_out,
        "x_after_ffn": x_after_ffn,
        "final": final,
        "logits_no_bias": logits_no_bias,
        "logits": logits,
    }


def argmax(values):
    return max(range(len(values)), key=lambda idx: values[idx])


def next_token(tokens, vocab, id_to_token, weights):
    input_ids = encode_prompt(tokens, vocab)
    result = tiny_lm_forward(input_ids, weights)
    token_id = argmax(result["logits"][-1])
    return token_id, id_to_token[token_id], result


def top_tokens(logits, id_to_token, limit=5):
    top_ids = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)[:limit]
    return [(idx, id_to_token[idx], logits[idx]) for idx in top_ids]


def pad_rows(rows, *, row_count=4, width=4):
    padded = []
    for row in rows[:row_count]:
        padded.append([lpu.to_f32(value) for value in row[:width]] + [0.0] * max(0, width - len(row)))
    while len(padded) < row_count:
        padded.append([0.0 for _ in range(width)])
    return padded


def fp8_quantize_matrix(matrix):
    bits = [[lpu.fp8_e5m2_bits(value) for value in row] for row in matrix]
    decoded = [[lpu.fp8_e5m2_to_f32(value) for value in row] for row in bits]
    return bits, decoded


def pack_fp32_row(row):
    word = 0
    for idx, value in enumerate(row):
        word |= lpu.f32_bits(value) << (32 * idx)
    return word


def unpack_fp8_word(word):
    return [(word >> (8 * idx)) & 0xFF for idx in range(4)]


def rope_rows_fp32(data, cos_bits, sin_bits):
    out = [0.0 for _ in range(4)]
    for pair in range(2):
        even = 2 * pair
        odd = even + 1
        x_even = lpu.to_f32(data[even])
        x_odd = lpu.to_f32(data[odd])
        cos_value = lpu.fp8_e5m2_to_f32(cos_bits[even])
        sin_value = lpu.fp8_e5m2_to_f32(sin_bits[even])
        out[even] = lpu.to_f32(
            lpu.to_f32(x_even * cos_value) - lpu.to_f32(x_odd * sin_value)
        )
        out[odd] = lpu.to_f32(
            lpu.to_f32(x_even * sin_value) + lpu.to_f32(x_odd * cos_value)
        )
    return out


def matrix_close(actual, expected, *, tol=1e-5):
    assert len(actual) == len(expected)
    assert len(actual[0]) == len(expected[0])
    for row_idx, row in enumerate(actual):
        for col_idx, value in enumerate(row):
            diff = abs(value - expected[row_idx][col_idx])
            assert diff <= tol, (
                f"matrix mismatch ({row_idx}, {col_idx}): "
                f"got {value:.8f}, expected {expected[row_idx][col_idx]:.8f}, diff {diff:.8f}"
            )


def mxm_expected_from_fp8_inputs(left_rows, right_rows):
    left_bits, left_decoded = fp8_quantize_matrix(pad_rows(left_rows))
    right_bits, right_decoded = fp8_quantize_matrix(pad_rows(right_rows))
    expected = lpu.matmul_expected_fp32(left_decoded, lpu.transpose_matrix(right_decoded))
    return left_bits, right_bits, expected


async def run_lpu_mxm_tile(
    dut,
    *,
    left_rows,
    right_rows,
    label,
    mem0_base=0,
    mem1_base=64,
):
    left_bits, right_bits, expected = mxm_expected_from_fp8_inputs(left_rows, right_rows)

    for k_idx in range(4):
        lpu.preload_mem0_word(
            dut,
            addr=mem0_base + k_idx,
            values=[left_bits[row_idx][k_idx] for row_idx in range(4)],
        )
        lpu.preload_mem1_word(
            dut,
            addr=mem1_base + k_idx,
            values=[right_bits[row_idx][k_idx] for row_idx in range(4)],
        )

    program = [lpu.build_instruction(mxm_clear=1, **FP_CTRL)]
    for k_idx in range(4):
        lpu.append_mxm_weight_row_load_from_mem1(program, addr=mem1_base + k_idx)
        lpu.append_mxm_input_column_load_from_mem0(program, addr=mem0_base + k_idx)
        program.append(lpu.build_instruction(mxm_start=1, **FP_CTRL))
        for _ in range(4):
            program.append(lpu.build_instruction(**FP_CTRL))
    program.extend([lpu.build_instruction(**FP_CTRL), lpu.build_instruction(**FP_CTRL)])

    await lpu.run_lpu_program(dut, program, extra_cycles=24)
    observed_bits = lpu.read_mxm_matrix_bits(dut)
    observed = [
        [lpu.bits_to_f32(observed_bits[row_idx][col_idx]) for col_idx in range(4)]
        for row_idx in range(4)
    ]

    for row_idx in range(4):
        for col_idx in range(4):
            expected_bits = lpu.f32_bits(expected[row_idx][col_idx])
            assert observed_bits[row_idx][col_idx] == expected_bits, (
                f"{label} tile ({row_idx}, {col_idx}) mismatch: "
                f"got 0x{observed_bits[row_idx][col_idx]:08x}, "
                f"expected 0x{expected_bits:08x}"
            )

    dut._log.info("%s MXM tile matched FP8->FP32 expected output", label)
    return observed


async def run_forced_vxm_row(
    dut,
    *,
    data,
    vxm_ctrl,
    fp_quant_mode=1,
    bias=None,
    gamma=None,
    beta=None,
    layernorm_en=0,
    rope_en=0,
    rope_cos_bits=None,
    rope_sin_bits=None,
    residual_op=lpu.VXM_RES_PASS,
    reset=True,
):
    if reset:
        await lpu.reset_dut(dut)
    else:
        dut.u_lpu.u_vxm.in_valid.value = Force(0)
        dut.u_lpu.u_vxm.out_ready.value = Force(1)
        for _ in range(8):
            if not int(dut.u_lpu.u_vxm.out_valid.value):
                break
            await lpu.tick(dut, 1)

    if bias is not None:
        dut.u_lpu.vxm_bias_reg.value = pack_fp32_row(bias)
    if gamma is not None:
        dut.u_lpu.vxm_layernorm_gamma_reg.value = pack_fp32_row(gamma)
    if beta is not None:
        dut.u_lpu.vxm_layernorm_beta_reg.value = pack_fp32_row(beta)
    if rope_cos_bits is not None:
        dut.u_lpu.vxm_rope_cos_fp8_reg.value = lpu.pack_bytes(rope_cos_bits)
    if rope_sin_bits is not None:
        dut.u_lpu.vxm_rope_sin_fp8_reg.value = lpu.pack_bytes(rope_sin_bits)

    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row(data))
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(vxm_ctrl)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(fp_quant_mode)
    dut.u_lpu.u_vxm.rope_en.value = Force(rope_en)
    dut.u_lpu.u_vxm.residual_op.value = Force(residual_op)
    dut.u_lpu.u_vxm.layernorm_bypass.value = Force(0 if layernorm_en else 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)

    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)

    for _ in range(160):
        await lpu.tick(dut, 1)
        if int(dut.u_lpu.u_vxm.out_valid.value):
            row_word = int(dut.u_lpu.u_vxm.stream_out.value) & 0xFFFFFFFF
            scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value) & 0xFFFFFFFF
            dut.u_lpu.u_vxm.stream_in_data.value = Release()
            dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
            dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
            dut.u_lpu.u_vxm.rope_en.value = Release()
            dut.u_lpu.u_vxm.residual_op.value = Release()
            dut.u_lpu.u_vxm.layernorm_bypass.value = Release()
            dut.u_lpu.u_vxm.in_valid.value = Release()
            dut.u_lpu.u_vxm.out_ready.value = Release()
            return row_word, scale_word

    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.layernorm_bypass.value = Release()
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    raise AssertionError("VXM did not produce an output row")


async def drive_forced_vxm_residual_op(
    dut,
    *,
    data,
    residual_op,
    reset=False,
):
    if reset:
        await lpu.reset_dut(dut)

    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row(data))
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(residual_op)
    dut.u_lpu.u_vxm.layernorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)

    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)

    for _ in range(120):
        await lpu.tick(dut, 1)
        if int(dut.u_lpu.u_vxm.residual_done.value):
            dut.u_lpu.u_vxm.stream_in_data.value = Release()
            dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
            dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
            dut.u_lpu.u_vxm.rope_en.value = Release()
            dut.u_lpu.u_vxm.residual_op.value = Release()
            dut.u_lpu.u_vxm.layernorm_bypass.value = Release()
            dut.u_lpu.u_vxm.in_valid.value = Release()
            dut.u_lpu.u_vxm.out_ready.value = Release()
            return

    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.layernorm_bypass.value = Release()
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    raise AssertionError("VXM residual op did not complete")


def token_rows_to_string(tokens):
    return " ".join(tokens)


@cocotb.test()
async def test_tiny_lm_prefill_decode_golden(dut):
    config, vocab, id_to_token, weights = load_tiny_lm_export()
    assert config == MODEL_CONFIG

    first_id, first_token, first = next_token(PROMPT_PREFILL, vocab, id_to_token, weights)
    second_id, second_token, second = next_token(
        PROMPT_PREFILL + [first_token],
        vocab,
        id_to_token,
        weights,
    )

    assert first_id == vocab["king"]
    assert second_id == vocab["."]
    assert first_id == argmax(first["logits"][-1])
    assert second_id == argmax(second["logits"][-1])

    dut._log.info('prefill prompt: "%s"', token_rows_to_string(PROMPT_PREFILL))
    dut._log.info("prefill next-token top5: %s", top_tokens(first["logits"][-1], id_to_token))
    dut._log.info('decode prompt: "%s"', token_rows_to_string(PROMPT_PREFILL + [first_token]))
    dut._log.info("decode next-token top5: %s", top_tokens(second["logits"][-1], id_to_token))
    dut._log.info('golden generation: "%s %s %s"', PROMPT_PREFILL[0], PROMPT_PREFILL[1], first_token)
    dut._log.info('next decoded token: "%s"', second_token)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_lpu_vxm_hardware_relu_softmax_layernorm_paths(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    relu_data = [-1.35, 0.49, 2.10, -0.75]
    relu_bias = [0.25, -0.90, 0.35, 1.40]
    relu_expected = [
        lpu.to_f32(max(0.0, lpu.to_f32(data + bias)))
        for data, bias in zip(relu_data, relu_bias)
    ]
    relu_expected_bits, relu_expected_scale = lpu.regular_fp8_row_quant_expected(relu_expected)
    relu_word, relu_scale = await run_forced_vxm_row(
        dut,
        data=relu_data,
        bias=relu_bias,
        vxm_ctrl=0b0011,
        fp_quant_mode=1,
    )
    assert unpack_fp8_word(relu_word) == relu_expected_bits
    assert (relu_scale & 0xFF) == (relu_expected_scale & 0xFF)
    dut._log.info("hardware VXM FP bias+ReLU quantized row: %s scale=%d", unpack_fp8_word(relu_word), relu_expected_scale)

    softmax_data = [0.35, -0.49, 1.20, -0.85]
    softmax_expected_bits = lpu.softmax_fp8_quant_expected(softmax_data)
    softmax_word, softmax_scale = await run_forced_vxm_row(
        dut,
        data=softmax_data,
        vxm_ctrl=0b1000,
        fp_quant_mode=1,
    )
    assert unpack_fp8_word(softmax_word) == softmax_expected_bits
    assert softmax_scale == 0
    dut._log.info("hardware VXM FP softmax quantized row: %s", unpack_fp8_word(softmax_word))

    ln_data = [0.35, -0.49, 1.25, -0.75]
    ln_gamma = [1.0, 0.5, 1.25, 0.75]
    ln_beta = [0.10, -0.20, 0.0, 0.35]
    ln_expected = layernorm_rows([ln_data], ln_gamma, ln_beta)[0]
    ln_expected_bits, ln_expected_scale = lpu.regular_fp8_row_quant_expected(ln_expected)
    ln_word, ln_scale = await run_forced_vxm_row(
        dut,
        data=ln_data,
        gamma=ln_gamma,
        beta=ln_beta,
        vxm_ctrl=0b0000,
        fp_quant_mode=1,
        layernorm_en=1,
    )
    assert unpack_fp8_word(ln_word) == ln_expected_bits
    assert (ln_scale & 0xFF) == (ln_expected_scale & 0xFF)
    dut._log.info("hardware VXM programmable layernorm quantized row: %s scale=%d", unpack_fp8_word(ln_word), ln_expected_scale)

    rope_data = [0.35, -0.49, 1.20, -0.85]
    cos_bits = [
        lpu.fp8_e5m2_bits(0.96),
        lpu.fp8_e5m2_bits(0.96),
        lpu.fp8_e5m2_bits(0.78),
        lpu.fp8_e5m2_bits(0.78),
    ]
    sin_bits = [
        lpu.fp8_e5m2_bits(0.29),
        lpu.fp8_e5m2_bits(0.29),
        lpu.fp8_e5m2_bits(-0.63),
        lpu.fp8_e5m2_bits(-0.63),
    ]

    lpu.preload_mem0_word(dut, addr=12, values=cos_bits)
    lpu.preload_mem1_word(dut, addr=18, values=sin_bits)
    load_rope_operands = [
        lpu.build_instruction(mem0_read_en=1, mem0_addr=12),
        lpu.build_instruction(
            eastbound_sel=lpu.EB_MEM0,
            eastbound_consumer_sel=lpu.EC_VXM,
            vxm_operand_sel=lpu.VXM_OPERAND_ROPE_COS,
        ),
        lpu.build_instruction(mem1_read_en=1, mem1_addr=18),
        lpu.build_instruction(
            westbound_sel=lpu.WB_MEM1,
            westbound_consumer_sel=lpu.WC_VXM,
            vxm_operand_sel=lpu.VXM_OPERAND_ROPE_SIN,
        ),
    ]
    await lpu.run_lpu_program(dut, load_rope_operands, extra_cycles=6)
    assert int(dut.u_lpu.vxm_rope_cos_fp8_reg.value) == lpu.pack_bytes(cos_bits)
    assert int(dut.u_lpu.vxm_rope_sin_fp8_reg.value) == lpu.pack_bytes(sin_bits)

    rope_expected = rope_rows_fp32(rope_data, cos_bits, sin_bits)
    rope_expected_bits, rope_expected_scale = lpu.regular_fp8_row_quant_expected(rope_expected)
    rope_word, rope_scale = await run_forced_vxm_row(
        dut,
        data=rope_data,
        vxm_ctrl=0b0000,
        fp_quant_mode=1,
        rope_en=1,
        rope_cos_bits=cos_bits,
        rope_sin_bits=sin_bits,
    )
    assert unpack_fp8_word(rope_word) == rope_expected_bits, (
        f"RoPE output mismatch: got {unpack_fp8_word(rope_word)}, "
        f"expected {rope_expected_bits}, "
        f"rope_out=0x{int(dut.u_lpu.u_vxm.rope_out.value):032x}, "
        f"rope_result=0x{int(dut.u_lpu.u_vxm.rope_result_reg.value):032x}, "
        f"rope_start={int(dut.u_lpu.u_vxm.rope_start.value)}, "
        f"rope_done={int(dut.u_lpu.u_vxm.rope_done.value)}, "
        f"rope_busy={int(dut.u_lpu.u_vxm.rope_busy.value)}, "
        f"rope_state={int(dut.u_lpu.u_vxm.rope_inst.state_q.value)}, "
        f"mux_valid={int(dut.u_lpu.u_vxm.mux_valid.value)}, "
        f"rope_result_valid={int(dut.u_lpu.u_vxm.rope_result_valid.value)}, "
        f"cos=0x{int(dut.u_lpu.vxm_rope_cos_fp8_reg.value):08x}, "
        f"sin=0x{int(dut.u_lpu.vxm_rope_sin_fp8_reg.value):08x}"
    )
    assert (rope_scale & 0xFF) == (rope_expected_scale & 0xFF)
    dut._log.info(
        "hardware VXM RoPE quantized row: %s scale=%d",
        unpack_fp8_word(rope_word),
        rope_expected_scale,
    )

    residual_base = [0.37, -0.82, 1.13, -1.41]
    residual_delta = [-0.29, 0.44, -0.61, 0.95]
    residual_expected = [
        lpu.to_f32(base + delta)
        for base, delta in zip(residual_base, residual_delta)
    ]
    residual_expected_bits, residual_expected_scale = lpu.regular_fp8_row_quant_expected(residual_expected)

    await drive_forced_vxm_residual_op(
        dut,
        data=residual_base,
        residual_op=lpu.VXM_RES_LOAD,
        reset=True,
    )
    await drive_forced_vxm_residual_op(
        dut,
        data=residual_delta,
        residual_op=lpu.VXM_RES_ADD,
    )
    residual_word, residual_scale = await run_forced_vxm_row(
        dut,
        data=[0.0, 0.0, 0.0, 0.0],
        vxm_ctrl=0b0000,
        fp_quant_mode=1,
        residual_op=lpu.VXM_RES_EMIT,
        reset=False,
    )
    assert unpack_fp8_word(residual_word) == residual_expected_bits
    assert (residual_scale & 0xFF) == (residual_expected_scale & 0xFF)
    dut._log.info(
        "hardware VXM residual add quantized row: %s scale=%d",
        unpack_fp8_word(residual_word),
        residual_expected_scale,
    )


@cocotb.test()
async def test_tiny_lm_ranvijay_prompt_probe(dut):
    config, vocab, id_to_token, weights = load_tiny_lm_export()
    assert config == MODEL_CONFIG

    token_id, token, result = next_token(PROMPT_PROBE, vocab, id_to_token, weights)
    assert token_id == argmax(result["logits"][-1])
    assert not token.startswith("<unused_")

    dut._log.info('probe prompt: "%s"', token_rows_to_string(PROMPT_PROBE))
    dut._log.info("probe next-token top10: %s", top_tokens(result["logits"][-1], id_to_token, limit=10))
    dut._log.info('probe generation: "%s %s"', token_rows_to_string(PROMPT_PROBE), token)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_lpu_ranvijay_prompt_timing(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    config, vocab, id_to_token, weights = load_tiny_lm_export()
    assert config == MODEL_CONFIG

    prompt_ids = encode_prompt(PROMPT_PROBE, vocab)
    golden = tiny_lm_forward(prompt_ids, weights)
    golden_token = id_to_token[argmax(golden["logits"][-1])]
    assert golden_token == "alpha"

    start_ns = get_sim_time(unit="ns")

    for projection, weight_name in [
        ("ranvijay Q projection", "blocks.0.attn.q_proj.weight"),
        ("ranvijay K projection", "blocks.0.attn.k_proj.weight"),
        ("ranvijay V projection", "blocks.0.attn.v_proj.weight"),
    ]:
        await run_lpu_mxm_tile(
            dut,
            left_rows=golden["ln1"],
            right_rows=weights[weight_name],
            label=projection,
        )

    await run_lpu_mxm_tile(
        dut,
        left_rows=golden["q"],
        right_rows=golden["k"],
        label="ranvijay causal attention Q @ K^T raw scores",
    )

    await run_lpu_mxm_tile(
        dut,
        left_rows=golden["probs"],
        right_rows=transpose(golden["v"]),
        label="ranvijay attention probabilities @ V",
    )

    await run_lpu_mxm_tile(
        dut,
        left_rows=golden["attn"],
        right_rows=weights["blocks.0.attn.out_proj.weight"],
        label="ranvijay attention output projection",
    )

    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        await run_lpu_mxm_tile(
            dut,
            left_rows=golden["ln2"],
            right_rows=weights["blocks.0.ffn.0.weight"][start:start + 4],
            label=f"ranvijay FFN W1 tile {start}:{start + 4}",
        )

    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        hidden_chunk = [row[start:start + 4] for row in golden["ffn_hidden"]]
        w2_chunk = [row[start:start + 4] for row in weights["blocks.0.ffn.2.weight"]]
        await run_lpu_mxm_tile(
            dut,
            left_rows=hidden_chunk,
            right_rows=w2_chunk,
            label=f"ranvijay FFN W2 partial tile {start}:{start + 4}",
        )

    last_hidden = golden["final"][-1]
    hw_logits = [0.0 for _ in range(MODEL_CONFIG["vocab_size"])]
    for vocab_start in range(0, MODEL_CONFIG["vocab_size"], 4):
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=[last_hidden],
            right_rows=weights["lm_head.weight"][vocab_start:vocab_start + 4],
            label=f"ranvijay LM head tile {vocab_start}:{vocab_start + 4}",
        )
        for lane in range(4):
            token_id = vocab_start + lane
            hw_logits[token_id] = lpu.to_f32(observed[0][lane] + weights["lm_head.bias"][token_id])

    end_ns = get_sim_time(unit="ns")
    elapsed_ns = end_ns - start_ns
    elapsed_cycles = elapsed_ns / 10.0
    hw_next_id = argmax(hw_logits)
    hw_next_token = id_to_token[hw_next_id]

    assert hw_next_token == golden_token
    assert hw_next_token == "alpha"

    dut._log.info('LPU-backed timing prompt: "%s"', token_rows_to_string(PROMPT_PROBE))
    dut._log.info("golden ranvijay top5: %s", top_tokens(golden["logits"][-1], id_to_token))
    dut._log.info("LPU-backed quantized ranvijay top5: %s", top_tokens(hw_logits, id_to_token))
    dut._log.info('LPU-backed next token for "%s" is "%s"', token_rows_to_string(PROMPT_PROBE), hw_next_token)
    dut._log.info("LPU-backed validation latency: %.0f ns = %.1f cycles at 10 ns clock", elapsed_ns, elapsed_cycles)


@cocotb.test()
async def test_lpu_tiny_lm_prefill_decode_tiles_and_lm_head(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    config, vocab, id_to_token, weights = load_tiny_lm_export()
    assert config == MODEL_CONFIG

    prefill_ids = encode_prompt(PROMPT_PREFILL, vocab)
    prefill = tiny_lm_forward(prefill_ids, weights)
    prefill_token = id_to_token[argmax(prefill["logits"][-1])]
    assert prefill_token == "king"

    decode_ids = encode_prompt(PROMPT_DECODE, vocab)
    decode = tiny_lm_forward(decode_ids, weights)
    decode_token = id_to_token[argmax(decode["logits"][-1])]
    assert decode_token == "."

    dut._log.info("LPU test uses prompt tokens: %s", PROMPT_DECODE)
    dut._log.info("Residual adds and causal mask are runtime/TB steps in this test.")
    dut._log.info("LPU MXM tiles execute the trained FP datapath matrix products.")

    for projection, weight_name in [
        ("Q projection", "blocks.0.attn.q_proj.weight"),
        ("K projection", "blocks.0.attn.k_proj.weight"),
        ("V projection", "blocks.0.attn.v_proj.weight"),
    ]:
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=decode["ln1"],
            right_rows=weights[weight_name],
            label=projection,
        )
        _, _, expected = mxm_expected_from_fp8_inputs(decode["ln1"], weights[weight_name])
        matrix_close(observed, expected)
        dut._log.info("%s hardware-compatible output rows: %s", projection, observed[:3])

    scores_raw = await run_lpu_mxm_tile(
        dut,
        left_rows=decode["q"],
        right_rows=decode["k"],
        label="causal attention Q @ K^T raw scores",
    )
    dut._log.info("raw QK scores before TB causal mask/scale: %s", [row[:3] for row in scores_raw[:3]])

    v_by_hidden = transpose(decode["v"])
    attn = await run_lpu_mxm_tile(
        dut,
        left_rows=decode["probs"],
        right_rows=v_by_hidden,
        label="attention probabilities @ V",
    )
    dut._log.info("attention @ V hardware-compatible rows: %s", attn[:3])

    observed = await run_lpu_mxm_tile(
        dut,
        left_rows=decode["attn"],
        right_rows=weights["blocks.0.attn.out_proj.weight"],
        label="attention output projection",
    )
    dut._log.info("attention output projection no-bias rows: %s", observed[:3])

    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=decode["ln2"],
            right_rows=weights["blocks.0.ffn.0.weight"][start:start + 4],
            label=f"FFN W1 tile {start}:{start + 4}",
        )
        dut._log.info("FFN W1 tile %d:%d no-bias rows: %s", start, start + 4, observed[:3])

    w2 = weights["blocks.0.ffn.2.weight"]
    ffn_w2_partials = []
    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        hidden_chunk = [row[start:start + 4] for row in decode["ffn_hidden"]]
        w2_chunk = [row[start:start + 4] for row in w2]
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=hidden_chunk,
            right_rows=w2_chunk,
            label=f"FFN W2 partial tile {start}:{start + 4}",
        )
        ffn_w2_partials.append(observed)
        dut._log.info("FFN W2 partial tile %d:%d rows: %s", start, start + 4, observed[:3])

    ffn_w2_sum = [[0.0 for _ in range(4)] for _ in range(3)]
    for partial in ffn_w2_partials:
        for row_idx in range(3):
            for col_idx in range(4):
                ffn_w2_sum[row_idx][col_idx] = lpu.to_f32(ffn_w2_sum[row_idx][col_idx] + partial[row_idx][col_idx])
    dut._log.info("summed FFN W2 no-bias hardware-compatible rows: %s", ffn_w2_sum)

    last_hidden = decode["final"][-1]
    hw_logits = [0.0 for _ in range(MODEL_CONFIG["vocab_size"])]
    for vocab_start in range(0, MODEL_CONFIG["vocab_size"], 4):
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=[last_hidden],
            right_rows=weights["lm_head.weight"][vocab_start:vocab_start + 4],
            label=f"LM head tile {vocab_start}:{vocab_start + 4}",
        )
        for lane in range(4):
            token_id = vocab_start + lane
            hw_logits[token_id] = lpu.to_f32(observed[0][lane] + weights["lm_head.bias"][token_id])

    hw_next_id = argmax(hw_logits)
    hw_next_token = id_to_token[hw_next_id]
    assert hw_next_token == "."

    dut._log.info("golden decode next-token top5: %s", top_tokens(decode["logits"][-1], id_to_token))
    dut._log.info("LPU-backed quantized LM-head top5: %s", top_tokens(hw_logits, id_to_token))
    dut._log.info('LPU-backed next token for "%s" is "%s"', token_rows_to_string(PROMPT_DECODE), hw_next_token)
