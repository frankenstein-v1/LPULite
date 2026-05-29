import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly, NextTimeStep
import random

# Global list to accumulate test records for the summary table
test_records = []

def pack_superlane(lanes: list[int]) -> int:
    assert len(lanes) == 4
    val = 0
    for j in range(4):
        val |= (lanes[j] & 0xFF) << (j * 8)
    return val

def unpack_superlane(val: int) -> list[int]:
    return [(int(val) >> (j * 8)) & 0xFF for j in range(4)]

def pack_opcode(opcodes: list[int]) -> int:
    assert len(opcodes) == 4
    val = 0
    for i in range(4):
        val |= (opcodes[i] & 0x7) << (i * 3)
    return val

class SxmReferenceModel:
    def __init__(self):
        self.reset()

    def reset(self):
        self.input_d1 = 0
        self.input_d2 = 0
        self.input_d3 = 0
        
        self.weight_d1 = 0
        self.weight_d2 = 0
        self.weight_d3 = 0

    def clock_edge(self, eastbound_in: int, westbound_in: int):
        # Update pipeline delays on positive edge
        self.input_d3 = self.input_d2
        self.input_d2 = self.input_d1
        self.input_d1 = eastbound_in
        
        self.weight_d3 = self.weight_d2
        self.weight_d2 = self.weight_d1
        self.weight_d1 = westbound_in

    def compute_outputs(self, eastbound_in: int, westbound_in: int, opcode_input: int, opcode_weight: int) -> tuple[int, int]:
        eb_lanes = unpack_superlane(eastbound_in)
        wb_lanes = unpack_superlane(westbound_in)
        
        id1_lanes = unpack_superlane(self.input_d1)
        id2_lanes = unpack_superlane(self.input_d2)
        id3_lanes = unpack_superlane(self.input_d3)
        
        wd1_lanes = unpack_superlane(self.weight_d1)
        wd2_lanes = unpack_superlane(self.weight_d2)
        wd3_lanes = unpack_superlane(self.weight_d3)
        
        eb_out = [0] * 4
        wb_out = [0] * 4
        
        for i in range(4):
            cur_op_in = (opcode_input >> (i * 3)) & 0x7
            cur_op_wt = (opcode_weight >> (i * 3)) & 0x7
            
            # Eastbound Router
            if cur_op_in == 0:
                eb_out[i] = eb_lanes[0]
            elif cur_op_in == 1:
                eb_out[i] = eb_lanes[1]
            elif cur_op_in == 2:
                eb_out[i] = eb_lanes[2]
            elif cur_op_in == 3:
                eb_out[i] = eb_lanes[3]
            elif cur_op_in == 4:
                eb_out[i] = id1_lanes[i]
            elif cur_op_in == 5:
                eb_out[i] = id2_lanes[i]
            elif cur_op_in == 6:
                eb_out[i] = id3_lanes[i]
            else:
                eb_out[i] = 0
                
            # Westbound Router
            if cur_op_wt == 0:
                wb_out[i] = wb_lanes[0]
            elif cur_op_wt == 1:
                wb_out[i] = wb_lanes[1]
            elif cur_op_wt == 2:
                wb_out[i] = wb_lanes[2]
            elif cur_op_wt == 3:
                wb_out[i] = wb_lanes[3]
            elif cur_op_wt == 4:
                wb_out[i] = wd1_lanes[i]
            elif cur_op_wt == 5:
                wb_out[i] = wd2_lanes[i]
            elif cur_op_wt == 6:
                wb_out[i] = wd3_lanes[i]
            else:
                wb_out[i] = 0
                
        return pack_superlane(eb_out), pack_superlane(wb_out)

def record_test(step_name: str, op_in: list[int], op_wt: list[int],
                eb_in: list[int], wb_in: list[int],
                eb_exp: list[int], wb_exp: list[int],
                eb_obs: list[int], wb_obs: list[int], passed: bool):
    test_records.append({
        "step": step_name,
        "op_in": op_in,
        "op_wt": op_wt,
        "eb_in": eb_in,
        "wb_in": wb_in,
        "eb_exp": eb_exp,
        "wb_exp": wb_exp,
        "eb_obs": eb_obs,
        "wb_obs": wb_obs,
        "passed": passed
    })

def print_summary_table(log_func):
    header = f"{'Test Case / Step':<28} | {'Op EB':<10} | {'Op WB':<10} | {'EB In':<14} | {'WB In':<14} | {'EB Out (Obs/Exp)':<20} | {'WB Out (Obs/Exp)':<20} | {'Status':<6}"
    sep = "=" * len(header)
    log_func(" ")
    log_func(sep)
    log_func("                             SXM VERIFICATION OUTCOMES SUMMARY")
    log_func(sep)
    log_func(header)
    log_func(sep)
    for r in test_records:
        eb_in_str = ",".join(f"{x:02X}" for x in r["eb_in"])
        wb_in_str = ",".join(f"{x:02X}" for x in r["wb_in"])
        eb_out_str = ",".join(f"{o:02X}" for o in r["eb_obs"]) + "/" + ",".join(f"{e:02X}" for e in r["eb_exp"])
        wb_out_str = ",".join(f"{o:02X}" for o in r["wb_obs"]) + "/" + ",".join(f"{e:02X}" for e in r["wb_exp"])
        op_in_str = ",".join(str(o) for o in r["op_in"])
        op_wt_str = ",".join(str(w) for w in r["op_wt"])
        status = "PASS" if r["passed"] else "FAIL"
        log_func(f"{r['step']:<28} | {op_in_str:<10} | {op_wt_str:<10} | {eb_in_str:<14} | {wb_in_str:<14} | {eb_out_str:<20} | {wb_out_str:<20} | {status:<6}")
    log_func(sep)
    log_func(" ")

async def reset_dut(dut, ref_model: SxmReferenceModel) -> None:
    dut.rst_n.value = 0
    dut.opcode_input.value = 0
    dut.opcode_weight.value = 0
    dut.eastbound_in.value = 0
    dut.westbound_in.value = 0
    
    ref_model.reset()
    
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await Timer(10, unit="ns")

async def check_step(dut, ref_model: SxmReferenceModel, step_name: str, 
                     eb_in_lanes: list[int], wb_in_lanes: list[int], 
                     op_in_lanes: list[int], op_wt_lanes: list[int]) -> bool:
    """Helper to apply inputs, compute expected, sample observed, and assert match."""
    eb_in_packed = pack_superlane(eb_in_lanes)
    wb_in_packed = pack_superlane(wb_in_lanes)
    op_in_packed = pack_opcode(op_in_lanes)
    op_wt_packed = pack_opcode(op_wt_lanes)
    
    # 1. Drive inputs
    dut.eastbound_in.value = eb_in_packed
    dut.westbound_in.value = wb_in_packed
    dut.opcode_input.value = op_in_packed
    dut.opcode_weight.value = op_wt_packed
    
    # 2. Wait for combinational logic to settle, then sample combinationally
    await Timer(1, unit="ns")
    
    eb_out_obs_val = int(dut.eastbound_out.value)
    wb_out_obs_val = int(dut.westbound_out.value)
    
    # Compute expected
    eb_out_exp_val, wb_out_exp_val = ref_model.compute_outputs(
        eb_in_packed, wb_in_packed, op_in_packed, op_wt_packed
    )
    
    eb_out_obs = unpack_superlane(eb_out_obs_val)
    wb_out_obs = unpack_superlane(wb_out_obs_val)
    
    eb_out_exp = unpack_superlane(eb_out_exp_val)
    wb_out_exp = unpack_superlane(wb_out_exp_val)
    
    passed = (eb_out_obs == eb_out_exp) and (wb_out_obs == wb_out_exp)
    
    record_test(step_name, op_in_lanes, op_wt_lanes, eb_in_lanes, wb_in_lanes,
                eb_out_exp, wb_out_exp, eb_out_obs, wb_out_obs, passed)
    
    assert passed, (
        f"Mismatch in {step_name}!\n"
        f"  Opcodes: EB={op_in_lanes}, WB={op_wt_lanes}\n"
        f"  Inputs : EB={eb_in_lanes}, WB={wb_in_lanes}\n"
        f"  EB Out expected {eb_out_exp}, got {eb_out_obs}\n"
        f"  WB Out expected {wb_out_exp}, got {wb_out_obs}"
    )
    return passed

@cocotb.test()
async def test_sxm_complete(dut):
    """End-to-end comprehensive test suite verifying the SXM hardware module."""
    # Start clock generator
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    
    ref = SxmReferenceModel()
    
    dut._log.info("Starting SXM (Systolic Exchange Module) Cocotb Test Suite...")
    
    # ==========================================
    # 1. Reset Behavior Test
    # ==========================================
    dut._log.info("1. Running Reset Behavior Test...")
    await reset_dut(dut, ref)
    # Check that outputs combinationally react under reset defaults (op = 0, so straight pass lane 0)
    # Since inputs are driven to 0, outputs should be 0.
    await check_step(dut, ref, "Reset Default State", 
                      [0xAA, 0xBB, 0xCC, 0xDD], [0x11, 0x22, 0x33, 0x44], 
                      [0, 0, 0, 0], [0, 0, 0, 0])
    
    # ==========================================
    # 2. Combinational Routing / Crossbars
    # ==========================================
    dut._log.info("2. Running Combinational Crossbar Tests...")
    
    # Straight pass: map output lane i to input lane i
    await check_step(dut, ref, "Comb Straight Pass",
                      [0x10, 0x20, 0x30, 0x40], [0x01, 0x02, 0x03, 0x04],
                      [0, 1, 2, 3], [0, 1, 2, 3])
                      
    # Reverse lanes: output lane i to input lane 3-i
    await check_step(dut, ref, "Comb Reverse Lanes",
                      [0x10, 0x20, 0x30, 0x40], [0x01, 0x02, 0x03, 0x04],
                      [3, 2, 1, 0], [3, 2, 1, 0])
                      
    # Broadcast lane 2
    await check_step(dut, ref, "Comb Broadcast Lane 2",
                      [0x10, 0x20, 0x30, 0x40], [0x01, 0x02, 0x03, 0x04],
                      [2, 2, 2, 2], [2, 2, 2, 2])
                      
    # Permutation mixed
    await check_step(dut, ref, "Comb Permutation Mixed",
                      [0xA, 0xB, 0xC, 0xD], [0xE, 0xF, 0x1, 0x2],
                      [1, 3, 0, 2], [3, 0, 2, 1])

    # ==========================================
    # 3. Systolic Delays (Synchronous Delay Lines)
    # ==========================================
    dut._log.info("3. Running Systolic Delay Line Tests...")
    
    # We will drive a sequence of inputs, clock them in, and then retrieve them via delay select codes (4, 5, 6)
    # Let's set some inputs and tick the clock
    inputs_seq = [
        ([0x11, 0x22, 0x33, 0x44], [0x55, 0x66, 0x77, 0x88]), # Step 1 (will be in D1 after 1 clock edge)
        ([0xAA, 0xBB, 0xCC, 0xDD], [0xEE, 0xFF, 0x12, 0x34]), # Step 2 (D1 has AA..DD, D2 has 11..44)
        ([0x99, 0x88, 0x77, 0x66], [0x55, 0x44, 0x33, 0x22])  # Step 3 (D1 has 99..66, D2 has AA..DD, D3 has 11..44)
    ]
    
    # Drive first input and check combinational straight pass first
    await check_step(dut, ref, "Systolic Init drive 1",
                      inputs_seq[0][0], inputs_seq[0][1],
                      [0, 1, 2, 3], [0, 1, 2, 3])
                      
    # Tick clock edge to shift the delay lines!
    ref.clock_edge(pack_superlane(inputs_seq[0][0]), pack_superlane(inputs_seq[0][1]))
    await RisingEdge(dut.clk)
    await NextTimeStep()
    
    # Drive second input and verify D1 output (op = 4 means delay 1)
    await check_step(dut, ref, "Systolic Delay 1-Cycle",
                      inputs_seq[1][0], inputs_seq[1][1],
                      [4, 4, 4, 4], [4, 4, 4, 4])
                      
    # Tick clock edge to shift again!
    ref.clock_edge(pack_superlane(inputs_seq[1][0]), pack_superlane(inputs_seq[1][1]))
    await RisingEdge(dut.clk)
    await NextTimeStep()
    
    # Drive third input and verify D2 output (op = 5 means delay 2)
    # The first sequence values should be at D2 now.
    await check_step(dut, ref, "Systolic Delay 2-Cycle",
                      inputs_seq[2][0], inputs_seq[2][1],
                      [5, 5, 5, 5], [5, 5, 5, 5])
                      
    # Tick clock edge to shift again!
    ref.clock_edge(pack_superlane(inputs_seq[2][0]), pack_superlane(inputs_seq[2][1]))
    await RisingEdge(dut.clk)
    await NextTimeStep()
    
    # Verify D3 output (op = 6 means delay 3)
    # The first sequence values should be at D3 now.
    await check_step(dut, ref, "Systolic Delay 3-Cycle",
                      [0x00, 0x00, 0x00, 0x00], [0x00, 0x00, 0x00, 0x00],
                      [6, 6, 6, 6], [6, 6, 6, 6])

    # ==========================================
    # 4. Bubble Output Test
    # ==========================================
    dut._log.info("4. Running Bubble Output Test...")
    # Opcode 7 (3'b111) should output zero
    await check_step(dut, ref, "Bubbles / Zero Outputs",
                      [0xFF, 0xFF, 0xFF, 0xFF], [0xFF, 0xFF, 0xFF, 0xFF],
                      [7, 7, 7, 7], [7, 7, 7, 7])

    # ==========================================
    # 5. Randomized Mixed Stress Test
    # ==========================================
    dut._log.info("5. Running Mixed Randomized Stress Test...")
    
    for cycle in range(50):
        # Random inputs
        eb_in = [random.randint(0, 255) for _ in range(4)]
        wb_in = [random.randint(0, 255) for _ in range(4)]
        # Random opcodes
        op_in = [random.randint(0, 7) for _ in range(4)]
        op_wt = [random.randint(0, 7) for _ in range(4)]
        
        # Verify state prior to tick
        await check_step(dut, ref, f"Random Step {cycle}", eb_in, wb_in, op_in, op_wt)
        
        # Tick the clock
        ref.clock_edge(pack_superlane(eb_in), pack_superlane(wb_in))
        await RisingEdge(dut.clk)
        await NextTimeStep()
        
    dut._log.info("SXM Cocotb Test Suite completed successfully! Printing summary table...")
    print_summary_table(dut._log.info)


@cocotb.test()
async def test_sxm_transpose(dut):
    """Verify that the SXM Transpose LOAD and EMIT functionality operates correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    
    dut._log.info("Starting SXM Transpose Mode Cocotb Test...")
    
    # 1. Reset
    dut.rst_n.value = 0
    dut.opcode_input.value = 0
    dut.opcode_weight.value = 0
    dut.eastbound_in.value = 0
    dut.westbound_in.value = 0
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await Timer(10, unit="ns")
    
    # 4 rows we want to load and transpose:
    input_matrix = [
        [0x01, 0x02, 0x03, 0x04], # Row 0
        [0x05, 0x06, 0x07, 0x08], # Row 1
        [0x09, 0x0A, 0x0B, 0x0C], # Row 2
        [0x0D, 0x0E, 0x0F, 0x10], # Row 3
    ]
    
    # Expected transposed rows:
    expected_transposed = [
        [0x01, 0x05, 0x09, 0x0D], # Col 0
        [0x02, 0x06, 0x0A, 0x0E], # Col 1
        [0x03, 0x07, 0x0B, 0x0F], # Col 2
        [0x04, 0x08, 0x0C, 0x10], # Col 3
    ]
    
    # 2. LOAD phase
    # Cycle 0: Trigger load with OP_TRANSPOSE_LOAD (12'h5A5) and drive first row
    dut.opcode_input.value = 0x5A5
    dut.eastbound_in.value = pack_superlane(input_matrix[0])
    
    await RisingEdge(dut.clk)
    await NextTimeStep()
    
    # Cycles 1, 2, 3: continue loading other rows
    for r in range(1, 4):
        dut.opcode_input.value = 0 # No trigger opcode needed anymore
        dut.eastbound_in.value = pack_superlane(input_matrix[r])
        await RisingEdge(dut.clk)
        await NextTimeStep()
        
    dut._log.info("SXM Transpose matrix loaded. Starting EMIT phase...")
    
    # 3. EMIT phase
    # Cycle 0: Trigger emit with OP_TRANSPOSE_EMIT (12'hA5A)
    dut.opcode_input.value = 0xA5A
    
    # Sample combinational output for row 0 immediately
    await Timer(1, unit="ns")
    obs_row0 = unpack_superlane(int(dut.eastbound_out.value))
    assert obs_row0 == expected_transposed[0], f"Transpose Row 0 mismatch: expected {expected_transposed[0]}, got {obs_row0}"
    
    await RisingEdge(dut.clk)
    await NextTimeStep()
    
    # Cycles 1, 2, 3: sample combinational outputs for other rows
    for c in range(1, 4):
        dut.opcode_input.value = 0 # No trigger opcode
        await Timer(1, unit="ns")
        obs_row = unpack_superlane(int(dut.eastbound_out.value))
        assert obs_row == expected_transposed[c], f"Transpose Row {c} mismatch: expected {expected_transposed[c]}, got {obs_row}"
        await RisingEdge(dut.clk)
        await NextTimeStep()
        
    dut._log.info("SXM Transpose verification passed successfully!")

