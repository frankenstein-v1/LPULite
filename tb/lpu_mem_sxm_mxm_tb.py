import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


WB_NONE = 0
WB_SXM = 1
WB_MEM0 = 2
WB_MEM1 = 4

WC_NONE = 0
WC_MXM = 1
WC_SXM = 2

INGRESS_NONE = 0
INGRESS_INPUT = 1
INGRESS_WGHT = 2


def pack_bytes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word


def signed_value(handle):
    return int(handle.value.to_signed())


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fp8_e5m2_bits(value: float) -> int:
    table = {
        0.0: 0x00,
        0.5: 0x38,
        1.0: 0x3C,
        2.0: 0x40,
        4.0: 0x44,
        8.0: 0x48,
        -0.5: 0xB8,
        -1.0: 0xBC,
        -2.0: 0xC0,
        -4.0: 0xC4,
        -8.0: 0xC8,
    }
    if value not in table:
        raise ValueError(f"unsupported FP8 test value {value}")
    return table[value]


def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)


def build_instruction(
    *,
    westbound_sel=WB_NONE,
    westbound_consumer_sel=WC_NONE,
    mem0_read_en=0,
    mem0_addr=0,
    mem1_read_en=0,
    mem1_addr=0,
    sxm_opcode_input=0,
    mxm_ingress_mode=INGRESS_NONE,
    mxm_start=0,
    mxm_clear=0,
    mxm_input_is_signed=1,
    mxm_wght_is_signed=1,
    mxm_use_fp=0,
):
    word = 0
    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_field(word, mem0_addr, 14, 9)
    word = _set_field(word, mem1_read_en, 23, 1)
    word = _set_field(word, mem1_addr, 25, 9)
    word = _set_field(word, sxm_opcode_input, 34, 12)
    word = _set_field(word, mxm_ingress_mode, 62, 2)
    word = _set_field(word, mxm_start, 64, 1)
    word = _set_field(word, mxm_clear, 65, 1)
    word = _set_field(word, mxm_input_is_signed, 77, 1)
    word = _set_field(word, mxm_wght_is_signed, 78, 1)
    word = _set_field(word, mxm_use_fp, 79, 1)
    return word


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.rst_n.value = 0
    await tick(dut, 2)
    dut.rst_n.value = 1
    await Timer(1, unit="ns")


def preload_instruction(dut, pc, instruction_word):
    dut.u_lpu.u_icu.imem_array[pc].value = instruction_word


def preload_mem0_word(dut, addr, values):
    dut.u_lpu.u_mem0.sram_array[addr].value = pack_bytes(values)


def preload_mem1_word(dut, addr, values):
    dut.u_lpu.u_mem1.sram_array[addr].value = pack_bytes(values)


def preload_program(dut, instructions, *, trailing_nops=32):
    for pc, instruction_word in enumerate(instructions):
        preload_instruction(dut, pc, instruction_word)
    for pc in range(len(instructions), len(instructions) + trailing_nops):
        preload_instruction(dut, pc, build_instruction())


def append_sxm_transpose_load_from_mem1(program, *, source_base):
    program.extend(
        [
            build_instruction(mem1_read_en=1, mem1_addr=source_base + 0),
            build_instruction(
                mem1_read_en=1,
                mem1_addr=source_base + 1,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(
                mem1_read_en=1,
                mem1_addr=source_base + 2,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
                sxm_opcode_input=0x5A5,
            ),
            build_instruction(
                mem1_read_en=1,
                mem1_addr=source_base + 3,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(),
        ]
    )


def read_mxm_matrix(dut):
    return [
        [signed_value(dut.mxm_out_00_dbg), signed_value(dut.mxm_out_01_dbg), signed_value(dut.mxm_out_02_dbg), signed_value(dut.mxm_out_03_dbg)],
        [signed_value(dut.mxm_out_10_dbg), signed_value(dut.mxm_out_11_dbg), signed_value(dut.mxm_out_12_dbg), signed_value(dut.mxm_out_13_dbg)],
        [signed_value(dut.mxm_out_20_dbg), signed_value(dut.mxm_out_21_dbg), signed_value(dut.mxm_out_22_dbg), signed_value(dut.mxm_out_23_dbg)],
        [signed_value(dut.mxm_out_30_dbg), signed_value(dut.mxm_out_31_dbg), signed_value(dut.mxm_out_32_dbg), signed_value(dut.mxm_out_33_dbg)],
    ]


@cocotb.test()
async def test_lpu_mem_sxm_mxm_qkt_single_k_slice(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    q_column0 = [3, -2, 5, 1]
    k_column0 = [7, 4, -3, 2]

    preload_mem0_word(dut, addr=0, values=q_column0)
    for row_idx, lane0_value in enumerate(k_column0):
        preload_mem1_word(dut, addr=row_idx, values=[lane0_value, 0, 0, 0])

    program = [
        build_instruction(mxm_clear=1),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0,
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
    ]
    append_sxm_transpose_load_from_mem1(program, source_base=0)
    program.extend(
        [
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                sxm_opcode_input=0xA5A,
            ),
            build_instruction(mxm_start=1),
            build_instruction(mxm_start=1),
            build_instruction(),
            build_instruction(),
        ]
    )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 6)

    assert int(dut.input_loaded_dbg.value) == 1
    assert int(dut.wght_loaded_dbg.value) == 1

    expected_scores = [
        [q_value * k_value for k_value in k_column0]
        for q_value in q_column0
    ]
    observed_scores = read_mxm_matrix(dut)
    assert observed_scores == expected_scores, (
        f"mem->sxm->mxm mismatch: got={observed_scores} expected={expected_scores}"
    )


@cocotb.test()
async def test_lpu_mem_sxm_mxm_qkt_single_k_slice_fp(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    q_column0 = [1.0, -0.5, 2.0, 0.5]
    k_column0 = [2.0, 0.5, -1.0, 4.0]

    preload_mem0_word(dut, addr=0, values=[fp8_e5m2_bits(v) for v in q_column0])
    for row_idx, lane0_value in enumerate(k_column0):
        preload_mem1_word(dut, addr=row_idx, values=[fp8_e5m2_bits(lane0_value), 0, 0, 0])

    fp_cfg = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)
    program = [
        build_instruction(mxm_clear=1, **fp_cfg),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0,
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
            **fp_cfg,
        ),
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
            **fp_cfg,
        ),
    ]
    append_sxm_transpose_load_from_mem1(program, source_base=0)
    program.extend(
        [
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                sxm_opcode_input=0xA5A,
                **fp_cfg,
            ),
            build_instruction(mxm_start=1, **fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
        ]
    )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 8)

    expected = [
        [q_value * k_value for k_value in k_column0]
        for q_value in q_column0
    ]
    observed_bits = [
        [int(dut.mxm_out_00_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_01_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_02_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_03_dbg.value) & 0xFFFFFFFF],
        [int(dut.mxm_out_10_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_11_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_12_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_13_dbg.value) & 0xFFFFFFFF],
        [int(dut.mxm_out_20_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_21_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_22_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_23_dbg.value) & 0xFFFFFFFF],
        [int(dut.mxm_out_30_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_31_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_32_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_33_dbg.value) & 0xFFFFFFFF],
    ]

    for r in range(4):
        for c in range(4):
            exp_bits = f32_bits(expected[r][c])
            got_bits = observed_bits[r][c]
            assert got_bits == exp_bits, (
                f"FP mem->sxm->mxm mismatch at ({r}, {c}): got 0x{got_bits:08x}, "
                f"expected 0x{exp_bits:08x}"
            )
