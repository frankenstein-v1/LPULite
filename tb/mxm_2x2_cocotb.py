import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def tick(dut, n=1):
    # Advance simulation by n clock cycles.
    # The extra 1ns delay after each rising edge gives combinational paths
    # and non-blocking assignments time to settle before checks.
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


@cocotb.test()
async def test_mxm_2x2_matmul(dut):
    """2x2 @ 2x2 matmul on mxm_size=2 wrapper."""

    # Start a free-running 10ns period clock on dut.clk.
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # Drive all DUT control/data inputs to a known default state.
    # Keep reset asserted first so internal state is deterministic.
    dut.rst.value = 1
    dut.mxm_clear.value = 0
    dut.mxm_start.value = 0
    dut.act0.value = 0
    dut.act1.value = 0
    dut.wght_load0.value = 0
    dut.wght_load1.value = 0
    dut.wght_val0.value = 0
    dut.wght_val1.value = 0

    # Hold reset for 2 cycles, then release it.
    await tick(dut, 2)
    dut.rst.value = 0

    # Explicitly clear accumulator state once after reset deassertion.
    dut.mxm_clear.value = 1
    await tick(dut, 1)
    dut.mxm_clear.value = 0

    # Define test matrices.
    # A = [[1,2],[3,4]]
    # B = [[5,6],[7,8]]
    # Expected C = A @ B = [[19,22],[43,50]]
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]

    # Enable MXM compute pipeline.
    dut.mxm_start.value = 1

    # Stream K dimension through the MXM.
    # For each k:
    # 1) load column weights B[k][0], B[k][1]
    # 2) drive activations A[0][k], A[1][k]
    # 3) tick once for load
    # 4) drop load and tick once for compute/accumulate
    for k in range(2):
        # Load phase: push k-th activation slice and k-th weight slice.
        dut.act0.value = a[0][k]
        dut.act1.value = a[1][k]
        dut.wght_val0.value = b[k][0]
        dut.wght_val1.value = b[k][1]
        dut.wght_load0.value = 1
        dut.wght_load1.value = 1
        await tick(dut, 1)

        # Compute phase: keep activations, stop loading new weights.
        dut.wght_load0.value = 0
        dut.wght_load1.value = 0
        await tick(dut, 1)

    # Flush one more cycle with zero activations so final pipeline state settles.
    dut.act0.value = 0
    dut.act1.value = 0
    await tick(dut, 1)

    # Stop computation after all required cycles are complete.
    dut.mxm_start.value = 0

    # Read final 2x2 output elements from flattened wrapper signals.
    c00 = int(dut.c00.value.signed_integer)
    c01 = int(dut.c01.value.signed_integer)
    c10 = int(dut.c10.value.signed_integer)
    c11 = int(dut.c11.value.signed_integer)

    # Check each matrix element against expected A @ B result.
    assert c00 == 19, f"c00 mismatch: got {c00}, expected 19"
    assert c01 == 22, f"c01 mismatch: got {c01}, expected 22"
    assert c10 == 43, f"c10 mismatch: got {c10}, expected 43"
    assert c11 == 50, f"c11 mismatch: got {c11}, expected 50"
