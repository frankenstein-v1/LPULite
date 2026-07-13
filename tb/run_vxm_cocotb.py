from pathlib import Path
import os
from cocotb_tools.runner import get_runner

def test_vxm():
    src_dir = (Path(__file__).parent / "../src").resolve()
    tb_dir = Path(__file__).parent.resolve()
    sim = os.getenv("SIM", "verilator")
    build_dir = (tb_dir / f"sim_build_vxm_{sim}").resolve()

    sources = [
        src_dir / "lut_layernorm.sv",
        src_dir / "cvfpu_fp32_addsub.sv",
        src_dir / "cvfpu_fp32_div.sv",
        src_dir / "cvfpu_fp32_cmp.sv",
        src_dir / "cvfpu_fp32_fma.sv",
        src_dir / "cvfpu_fp8_to_fp32_cast.sv",
        src_dir / "vxm_rope.sv",
        src_dir / "residual_add.sv",
        src_dir / "vxm.sv",
        src_dir / "softmax.sv",
        src_dir / "quant.sv",
        src_dir / "lut_softmax_exp.sv",
        src_dir / "lut_softmax_div.sv"
    ]

    runner = get_runner(sim)

    build_kwargs = dict(
        sources=sources,
        hdl_toplevel="vxm",
        always=True,
        build_dir=build_dir,
    )

    if sim == "verilator":
        cvfpu = (tb_dir / "../third_party/cvfpu").resolve()
        common = cvfpu / "src/common_cells/src"
        divsqrt = cvfpu / "src/fpu_div_sqrt_mvp/hdl"

        cvfpu_sources = [
            common / "cf_math_pkg.sv",
            common / "lzc.sv",
            common / "rr_arb_tree.sv",
            divsqrt / "defs_div_sqrt_mvp.sv",
            divsqrt / "iteration_div_sqrt_mvp.sv",
            divsqrt / "control_mvp.sv",
            divsqrt / "norm_div_sqrt_mvp.sv",
            divsqrt / "preprocess_mvp.sv",
            divsqrt / "nrbd_nrsc_mvp.sv",
            divsqrt / "div_sqrt_top_mvp.sv",
            divsqrt / "div_sqrt_mvp_wrapper.sv",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v",
            cvfpu / "vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ctrl.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_double.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ff1.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_pack.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_prepare.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_round.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_scalar_dp.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_bound_table.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_with_sqrt.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt.v",
            cvfpu / "vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_top.v",
            cvfpu / "vendor/cvw/fma/fmalza.sv",
            cvfpu / "src/fpnew_pkg.sv",
            cvfpu / "src/fpnew_cast_multi.sv",
            cvfpu / "src/fpnew_classifier.sv",
            cvfpu / "src/fpnew_divsqrt_th_32.sv",
            cvfpu / "src/fpnew_divsqrt_th_64_multi.sv",
            cvfpu / "src/fpnew_divsqrt_multi.sv",
            cvfpu / "src/fpnew_fma.sv",
            cvfpu / "src/fpnew_fma_multi.sv",
            cvfpu / "src/fpnew_noncomp.sv",
            cvfpu / "src/fpnew_opgroup_block.sv",
            cvfpu / "src/fpnew_opgroup_fmt_slice.sv",
            cvfpu / "src/fpnew_opgroup_multifmt_slice.sv",
            cvfpu / "src/fpnew_rounding.sv",
            cvfpu / "src/fpnew_top.sv",
        ]
        build_kwargs["sources"] = cvfpu_sources + sources
        build_kwargs["includes"] = [src_dir, cvfpu / "src/common_cells/include"]
        build_kwargs["defines"] = {"HAVE_CVFPU": 1}
        build_kwargs["build_args"] = ["--Wno-fatal"]
        build_kwargs["waves"] = True
    
    runner.build(**build_kwargs)

    runner.test(
        hdl_toplevel="vxm",
        test_module="vxm_tb",
        testcase=os.getenv("TESTCASE"),
        build_dir=build_dir,
        waves=(sim == "verilator"),
    )

if __name__ == "__main__":
    test_vxm()
