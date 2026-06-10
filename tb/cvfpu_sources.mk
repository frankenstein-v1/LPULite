CVFPU_COMMON_SOURCES = \
	$(COMMON)/cf_math_pkg.sv \
	$(COMMON)/lzc.sv \
	$(COMMON)/rr_arb_tree.sv

CVFPU_DIVSQRT_SOURCES = \
	$(DIVSQRT)/defs_div_sqrt_mvp.sv \
	$(DIVSQRT)/iteration_div_sqrt_mvp.sv \
	$(DIVSQRT)/control_mvp.sv \
	$(DIVSQRT)/norm_div_sqrt_mvp.sv \
	$(DIVSQRT)/preprocess_mvp.sv \
	$(DIVSQRT)/nrbd_nrsc_mvp.sv \
	$(DIVSQRT)/div_sqrt_top_mvp.sv \
	$(DIVSQRT)/div_sqrt_mvp_wrapper.sv

CVFPU_VENDOR_SOURCES = \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v \
	$(CVFPU)/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ctrl.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_double.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ff1.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_pack.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_prepare.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_round.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_scalar_dp.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_bound_table.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_with_sqrt.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt.v \
	$(CVFPU)/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_top.v \
	$(CVFPU)/vendor/cvw/fma/fmalza.sv

CVFPU_CORE_SOURCES = \
	$(CVFPU)/src/fpnew_pkg.sv \
	$(CVFPU)/src/fpnew_cast_multi.sv \
	$(CVFPU)/src/fpnew_classifier.sv \
	$(CVFPU)/src/fpnew_divsqrt_th_32.sv \
	$(CVFPU)/src/fpnew_divsqrt_th_64_multi.sv \
	$(CVFPU)/src/fpnew_divsqrt_multi.sv \
	$(CVFPU)/src/fpnew_fma.sv \
	$(CVFPU)/src/fpnew_fma_multi.sv \
	$(CVFPU)/src/fpnew_noncomp.sv \
	$(CVFPU)/src/fpnew_opgroup_block.sv \
	$(CVFPU)/src/fpnew_opgroup_fmt_slice.sv \
	$(CVFPU)/src/fpnew_opgroup_multifmt_slice.sv \
	$(CVFPU)/src/fpnew_rounding.sv \
	$(CVFPU)/src/fpnew_top.sv

CVFPU_ALL_SOURCES = \
	$(CVFPU_COMMON_SOURCES) \
	$(CVFPU_DIVSQRT_SOURCES) \
	$(CVFPU_VENDOR_SOURCES) \
	$(CVFPU_CORE_SOURCES)
