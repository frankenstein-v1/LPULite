import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


MXM_SIZE = 8
INPUT_SCALE = -2
WEIGHT_SCALE = -2
OUTPUT_SCALE = INPUT_SCALE + WEIGHT_SCALE

A_RAW = [
    [1, 2, -1, 0, 3, -2, 1, 4],
    [0, -1, 2, 1, -3, 2, 5, -1],
    [3, 0, 1, -2, 2, 1, -1, 0],
    [-2, 4, 0, 3, 1, -1, 2, 1],
    [1, -3, 2, 2, 0, 4, -2, 3],
    [2, 1, -4, 0, 1, 3, 2, -2],
    [0, 2, 3, -1, 4, -3, 1, 2],
    [-1, 0, 2, 5, -2, 1, 3, -4],
]

B_RAW = [
    [2, 0, -1, 3, 1, -2, 4, 1],
    [-1, 3, 2, 0, 1, 1, -2, 2],
    [0, -2, 1, 4, -3, 2, 1, -1],
    [3, 1, 0, -2, 2, -1, 1, 4],
    [1, -1, 3, 2, 0, 4, -3, 2],
    [-2, 2, -1, 1, 3, 0, 2, -4],
    [4, -3, 2, -1, 1, 3, 0, 1],
    [1, 2, -2, 3, -1, 1, 4, 0],
]


def scaled_real(raw, scale):
    return raw * (2.0 ** scale)


def pack_i8_lanes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word


def signed32(value):
    value = int(value) & 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def unpack_result_matrix(flat_value):
    raw = int(flat_value)
    matrix = []
    for row in range(MXM_SIZE):
        row_values = []
        for col in range(MXM_SIZE):
            shift = 32 * (MXM_SIZE * row + col)
            row_values.append(signed32(raw >> shift))
        matrix.append(row_values)
    return matrix


def matmul_expected(a, b):
    return [
        [
            sum(a[row][k] * b[k][col] for k in range(MXM_SIZE))
            for col in range(MXM_SIZE)
        ]
        for row in range(MXM_SIZE)
    ]


def decode_matrix(raw_matrix, scale):
    return [[scaled_real(value, scale) for value in row] for row in raw_matrix]


def format_int_matrix(matrix):
    return "\n".join("    " + " ".join(f"{value:5d}" for value in row) for row in matrix)


def format_decimal_matrix(matrix):
    return "\n".join("    " + " ".join(f"{value:7.4f}" for value in row) for row in matrix)


async def tick(dut, count=1):
    for _ in range(count):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.rst.value = 1
    dut.mxm_clear.value = 0
    dut.mxm_start.value = 0
    dut.input_vec.value = 0
    dut.input_scale.value = 0
    dut.weight_vec.value = 0
    dut.weight_scale.value = 0
    dut.wght_load.value = 0
    await tick(dut, 2)

    dut.rst.value = 0
    dut.mxm_clear.value = 1
    await tick(dut, 1)
    dut.mxm_clear.value = 0
    await tick(dut, 1)


async def apply_outer_product_step(dut, a_col, b_row):
    dut.input_vec.value = pack_i8_lanes(a_col)
    dut.input_scale.value = INPUT_SCALE
    dut.weight_vec.value = pack_i8_lanes(b_row)
    dut.weight_scale.value = WEIGHT_SCALE

    dut.wght_load.value = 0xFF
    dut.mxm_start.value = 0
    await tick(dut, 1)

    dut.wght_load.value = 0
    dut.mxm_start.value = 1
    await tick(dut, 1)

    dut.mxm_start.value = 0
    await tick(dut, 2)


@cocotb.test()
async def test_mxm_8x8_signed_matrix_multiply(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for k in range(MXM_SIZE):
        a_col = [A_RAW[row][k] for row in range(MXM_SIZE)]
        b_row = B_RAW[k]
        await apply_outer_product_step(dut, a_col, b_row)

    observed_raw = unpack_result_matrix(dut.mxm_out_flat.value)
    expected_raw = matmul_expected(A_RAW, B_RAW)
    observed_scale = int(dut.mxm_out_scale.value.to_signed())
    observed_decimal = decode_matrix(observed_raw, observed_scale)
    expected_decimal = decode_matrix(expected_raw, OUTPUT_SCALE)

    dut._log.info("A raw matrix, scale=%d:\n%s", INPUT_SCALE, format_int_matrix(A_RAW))
    dut._log.info("A decimal matrix:\n%s", format_decimal_matrix(decode_matrix(A_RAW, INPUT_SCALE)))
    dut._log.info("B raw matrix, scale=%d:\n%s", WEIGHT_SCALE, format_int_matrix(B_RAW))
    dut._log.info("B decimal matrix:\n%s", format_decimal_matrix(decode_matrix(B_RAW, WEIGHT_SCALE)))
    dut._log.info("Expected raw A*B:\n%s", format_int_matrix(expected_raw))
    dut._log.info(
        "Expected decimal A*B, scale=%d:\n%s",
        OUTPUT_SCALE,
        format_decimal_matrix(expected_decimal),
    )
    dut._log.info(
        "Observed MXM raw result:\n%s",
        format_int_matrix(observed_raw),
    )
    dut._log.info(
        "Observed MXM decimal result, scale=%d:\n%s",
        observed_scale,
        format_decimal_matrix(observed_decimal),
    )

    assert observed_raw == expected_raw
    assert observed_scale == OUTPUT_SCALE
    assert observed_decimal == expected_decimal
