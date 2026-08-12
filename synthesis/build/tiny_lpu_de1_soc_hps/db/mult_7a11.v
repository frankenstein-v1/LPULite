//lpm_mult CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEDICATED_MULTIPLIER_CIRCUITRY="NO" DEVICE_FAMILY="Cyclone V" DSP_BLOCK_BALANCING="Logic ELements" INPUT_A_IS_CONSTANT="NO" INPUT_B_IS_CONSTANT="NO" LPM_REPRESENTATION="SIGNED" LPM_WIDTHA=9 LPM_WIDTHB=9 LPM_WIDTHP=18 LPM_WIDTHS=1 MAXIMIZE_SPEED=5 dataa datab result CARRY_CHAIN="MANUAL" CARRY_CHAIN_LENGTH=48
//VERSION_BEGIN 25.1 cbx_cycloneii 2025:10:22:10:31:27:SC cbx_lpm_add_sub 2025:10:22:10:31:27:SC cbx_lpm_mult 2025:10:22:10:31:26:SC cbx_mgl 2025:10:22:10:31:44:SC cbx_nadder 2025:10:22:10:31:27:SC cbx_padd 2025:10:22:10:31:26:SC cbx_stratix 2025:10:22:10:31:27:SC cbx_stratixii 2025:10:22:10:31:26:SC cbx_util_mgl 2025:10:22:10:31:27:SC  VERSION_END
// synthesis VERILOG_INPUT_VERSION VERILOG_2001
// altera message_off 10463



// Copyright (C) 2025  Altera Corporation. All rights reserved.
//  Your use of Altera Corporation's design tools, logic functions 
//  and other software and tools, and any partner logic 
//  functions, and any output files from any of the foregoing 
//  (including device programming or simulation files), and any 
//  associated documentation or information are expressly subject 
//  to the terms and conditions of the Altera Program License 
//  Subscription Agreement, the Altera Quartus Prime License Agreement,
//  the Altera IP License Agreement, or other applicable license
//  agreement, including, without limitation, that your use is for
//  the sole purpose of programming logic devices manufactured by
//  Altera and sold by Altera or its authorized distributors.  Please
//  refer to the Altera Software License Subscription Agreements 
//  on the Quartus Prime software download page.




//adder CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" WIDTH=5 dataa datab datac result
//VERSION_BEGIN 25.1 cbx_mgl 2025:10:22:10:31:44:SC cbx_stratixii 2025:10:22:10:31:26:SC cbx_util_mgl 2025:10:22:10:31:27:SC  VERSION_END


//adder CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" WIDTH=17 dataa datab datac result
//VERSION_BEGIN 25.1 cbx_mgl 2025:10:22:10:31:44:SC cbx_stratixii 2025:10:22:10:31:26:SC cbx_util_mgl 2025:10:22:10:31:27:SC  VERSION_END


//adder CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" WIDTH=12 dataa datab datac result
//VERSION_BEGIN 25.1 cbx_mgl 2025:10:22:10:31:44:SC cbx_stratixii 2025:10:22:10:31:26:SC cbx_util_mgl 2025:10:22:10:31:27:SC  VERSION_END


//lpm_add_sub CARRY_CHAIN="MANUAL" CARRY_CHAIN_LENGTH=48 CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" LPM_REPRESENTATION="SIGNED" LPM_WIDTH=10 dataa datab result
//VERSION_BEGIN 25.1 cbx_cycloneii 2025:10:22:10:31:27:SC cbx_lpm_add_sub 2025:10:22:10:31:27:SC cbx_mgl 2025:10:22:10:31:44:SC cbx_nadder 2025:10:22:10:31:27:SC cbx_stratix 2025:10:22:10:31:27:SC cbx_stratixii 2025:10:22:10:31:26:SC  VERSION_END


//lpm_add_sub CARRY_CHAIN="MANUAL" CARRY_CHAIN_LENGTH=48 CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" LPM_REPRESENTATION="SIGNED" LPM_WIDTH=10 dataa datab result
//VERSION_BEGIN 25.1 cbx_cycloneii 2025:10:22:10:31:27:SC cbx_lpm_add_sub 2025:10:22:10:31:27:SC cbx_mgl 2025:10:22:10:31:44:SC cbx_nadder 2025:10:22:10:31:27:SC cbx_stratix 2025:10:22:10:31:27:SC cbx_stratixii 2025:10:22:10:31:26:SC  VERSION_END


//lpm_add_sub CARRY_CHAIN="MANUAL" CARRY_CHAIN_LENGTH=48 CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" LPM_REPRESENTATION="SIGNED" LPM_WIDTH=10 dataa datab result
//VERSION_BEGIN 25.1 cbx_cycloneii 2025:10:22:10:31:27:SC cbx_lpm_add_sub 2025:10:22:10:31:27:SC cbx_mgl 2025:10:22:10:31:44:SC cbx_nadder 2025:10:22:10:31:27:SC cbx_stratix 2025:10:22:10:31:27:SC cbx_stratixii 2025:10:22:10:31:26:SC  VERSION_END


//lpm_add_sub CARRY_CHAIN="MANUAL" CARRY_CHAIN_LENGTH=48 CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Cyclone V" LPM_REPRESENTATION="SIGNED" LPM_WIDTH=10 dataa datab result
//VERSION_BEGIN 25.1 cbx_cycloneii 2025:10:22:10:31:27:SC cbx_lpm_add_sub 2025:10:22:10:31:27:SC cbx_mgl 2025:10:22:10:31:44:SC cbx_nadder 2025:10:22:10:31:27:SC cbx_stratix 2025:10:22:10:31:27:SC cbx_stratixii 2025:10:22:10:31:26:SC  VERSION_END

//synthesis_resources = lut 108 
//synopsys translate_off
`timescale 1 ps / 1 ps
//synopsys translate_on
module  mult_7a11
	( 
	dataa,
	datab,
	result) /* synthesis synthesis_clearbox=1 */;
	input   [8:0]  dataa;
	input   [8:0]  datab;
	output   [17:0]  result;

	wire	[4:0]	wire_add12_result;
	wire	[16:0]	wire_add2_result;
	wire	[11:0]	wire_add7_result;
	wire	[9:0]	wire_sum_adder1aa_0dataa;
	wire	[9:0]	wire_sum_adder1aa_0datab;
	wire	[9:0]	wire_sum_adder1aa_0result;
	wire	[9:0]	wire_sum_adder1aa_1dataa;
	wire	[9:0]	wire_sum_adder1aa_1datab;
	wire	[9:0]	wire_sum_adder1aa_1result;
	wire	[9:0]	wire_sum_adder1aa_2dataa;
	wire	[9:0]	wire_sum_adder1aa_2datab;
	wire	[9:0]	wire_sum_adder1aa_2result;
	wire	[9:0]	wire_sum_adder1aa_3dataa;
	wire	[9:0]	wire_sum_adder1aa_3datab;
	wire	[9:0]	wire_sum_adder1aa_3result;
	wire  [11:0]   wire_sft10a_in;
	wire  [11:0]   wire_sft10a_out;
	wire  [11:0]   wire_sft11a_in;
	wire  [11:0]   wire_sft11a_out;
	wire  [4:0]   wire_sft13a_in;
	wire  [4:0]   wire_sft13a_out;
	wire  [4:0]   wire_sft14a_in;
	wire  [4:0]   wire_sft14a_out;
	wire  [4:0]   wire_sft15a_in;
	wire  [4:0]   wire_sft15a_out;
	wire  [4:0]   wire_sft16a_in;
	wire  [4:0]   wire_sft16a_out;
	wire  [16:0]   wire_sft3a_in;
	wire  [16:0]   wire_sft3a_out;
	wire  [16:0]   wire_sft4a_in;
	wire  [16:0]   wire_sft4a_out;
	wire  [16:0]   wire_sft5a_in;
	wire  [16:0]   wire_sft5a_out;
	wire  [16:0]   wire_sft6a_in;
	wire  [16:0]   wire_sft6a_out;
	wire  [11:0]   wire_sft8a_in;
	wire  [11:0]   wire_sft8a_out;
	wire  [11:0]   wire_sft9a_in;
	wire  [11:0]   wire_sft9a_out;
	wire  [8:0]  dataa_node;
	wire  [8:0]  datab_node;
	wire  [17:0]  final_result_node;
	wire  w192w;
	wire  w205w;
	wire  [18:0]  w339w;
	wire  [80:0]  w_decoder_node9w;
	wire  [8:0]  w_le_datab_node8w;
	wire  [8:0]  w_lone_sum_node11w;
	wire  [43:0]  w_sum_node10w;

	assign
		wire_add12_result = wire_sft13a_out + wire_sft14a_out + wire_sft15a_out;
	assign
		wire_add2_result = wire_sft3a_out + wire_sft4a_out + wire_sft5a_out;
	assign
		wire_add7_result = wire_sft8a_out + wire_sft9a_out + wire_sft10a_out;
	assign
		wire_sum_adder1aa_0result = wire_sum_adder1aa_0dataa + wire_sum_adder1aa_0datab;
	assign
		wire_sum_adder1aa_0dataa = {w_decoder_node9w[17], w_decoder_node9w[17:9]},
		wire_sum_adder1aa_0datab = {{2{w_decoder_node9w[8]}}, w_decoder_node9w[8:1]};
	assign
		wire_sum_adder1aa_1result = wire_sum_adder1aa_1dataa + wire_sum_adder1aa_1datab;
	assign
		wire_sum_adder1aa_1dataa = {w_decoder_node9w[35], w_decoder_node9w[35:27]},
		wire_sum_adder1aa_1datab = {{2{w_decoder_node9w[26]}}, w_decoder_node9w[26:19]};
	assign
		wire_sum_adder1aa_2result = wire_sum_adder1aa_2dataa + wire_sum_adder1aa_2datab;
	assign
		wire_sum_adder1aa_2dataa = {w_decoder_node9w[53], w_decoder_node9w[53:45]},
		wire_sum_adder1aa_2datab = {{2{w_decoder_node9w[44]}}, w_decoder_node9w[44:37]};
	assign
		wire_sum_adder1aa_3result = wire_sum_adder1aa_3dataa + wire_sum_adder1aa_3datab;
	assign
		wire_sum_adder1aa_3dataa = {w_decoder_node9w[71], w_decoder_node9w[71:63]},
		wire_sum_adder1aa_3datab = {{2{w_decoder_node9w[62]}}, w_decoder_node9w[62:55]};
	soft   sft10a_0
	( 
	.in(wire_sft10a_in[0:0]),
	.out(wire_sft10a_out[0:0]));
	soft   sft10a_1
	( 
	.in(wire_sft10a_in[1:1]),
	.out(wire_sft10a_out[1:1]));
	soft   sft10a_2
	( 
	.in(wire_sft10a_in[2:2]),
	.out(wire_sft10a_out[2:2]));
	soft   sft10a_3
	( 
	.in(wire_sft10a_in[3:3]),
	.out(wire_sft10a_out[3:3]));
	soft   sft10a_4
	( 
	.in(wire_sft10a_in[4:4]),
	.out(wire_sft10a_out[4:4]));
	soft   sft10a_5
	( 
	.in(wire_sft10a_in[5:5]),
	.out(wire_sft10a_out[5:5]));
	soft   sft10a_6
	( 
	.in(wire_sft10a_in[6:6]),
	.out(wire_sft10a_out[6:6]));
	soft   sft10a_7
	( 
	.in(wire_sft10a_in[7:7]),
	.out(wire_sft10a_out[7:7]));
	soft   sft10a_8
	( 
	.in(wire_sft10a_in[8:8]),
	.out(wire_sft10a_out[8:8]));
	soft   sft10a_9
	( 
	.in(wire_sft10a_in[9:9]),
	.out(wire_sft10a_out[9:9]));
	soft   sft10a_10
	( 
	.in(wire_sft10a_in[10:10]),
	.out(wire_sft10a_out[10:10]));
	soft   sft10a_11
	( 
	.in(wire_sft10a_in[11:11]),
	.out(wire_sft10a_out[11:11]));
	assign
		wire_sft10a_in = {w205w, {w205w, {w205w, {w205w, {w205w, {(~ w_sum_node10w[21]), {w192w, {(~ w_lone_sum_node11w[2]), {(~ w_lone_sum_node11w[1]), {(~ w_lone_sum_node11w[0]), {2{w205w}}}}}}}}}}}};
	soft   sft11a_0
	( 
	.in(wire_sft11a_in[0:0]),
	.out(wire_sft11a_out[0:0]));
	soft   sft11a_1
	( 
	.in(wire_sft11a_in[1:1]),
	.out(wire_sft11a_out[1:1]));
	soft   sft11a_2
	( 
	.in(wire_sft11a_in[2:2]),
	.out(wire_sft11a_out[2:2]));
	soft   sft11a_3
	( 
	.in(wire_sft11a_in[3:3]),
	.out(wire_sft11a_out[3:3]));
	soft   sft11a_4
	( 
	.in(wire_sft11a_in[4:4]),
	.out(wire_sft11a_out[4:4]));
	soft   sft11a_5
	( 
	.in(wire_sft11a_in[5:5]),
	.out(wire_sft11a_out[5:5]));
	soft   sft11a_6
	( 
	.in(wire_sft11a_in[6:6]),
	.out(wire_sft11a_out[6:6]));
	soft   sft11a_7
	( 
	.in(wire_sft11a_in[7:7]),
	.out(wire_sft11a_out[7:7]));
	soft   sft11a_8
	( 
	.in(wire_sft11a_in[8:8]),
	.out(wire_sft11a_out[8:8]));
	soft   sft11a_9
	( 
	.in(wire_sft11a_in[9:9]),
	.out(wire_sft11a_out[9:9]));
	soft   sft11a_10
	( 
	.in(wire_sft11a_in[10:10]),
	.out(wire_sft11a_out[10:10]));
	soft   sft11a_11
	( 
	.in(wire_sft11a_in[11:11]),
	.out(wire_sft11a_out[11:11]));
	assign
		wire_sft11a_in = wire_add7_result;
	soft   sft13a_0
	( 
	.in(wire_sft13a_in[0:0]),
	.out(wire_sft13a_out[0:0]));
	soft   sft13a_1
	( 
	.in(wire_sft13a_in[1:1]),
	.out(wire_sft13a_out[1:1]));
	soft   sft13a_2
	( 
	.in(wire_sft13a_in[2:2]),
	.out(wire_sft13a_out[2:2]));
	soft   sft13a_3
	( 
	.in(wire_sft13a_in[3:3]),
	.out(wire_sft13a_out[3:3]));
	soft   sft13a_4
	( 
	.in(wire_sft13a_in[4:4]),
	.out(wire_sft13a_out[4:4]));
	assign
		wire_sft13a_in = {w205w, {w_sum_node10w[20], {w192w, {w_sum_node10w[9], w192w}}}};
	soft   sft14a_0
	( 
	.in(wire_sft14a_in[0:0]),
	.out(wire_sft14a_out[0:0]));
	soft   sft14a_1
	( 
	.in(wire_sft14a_in[1:1]),
	.out(wire_sft14a_out[1:1]));
	soft   sft14a_2
	( 
	.in(wire_sft14a_in[2:2]),
	.out(wire_sft14a_out[2:2]));
	soft   sft14a_3
	( 
	.in(wire_sft14a_in[3:3]),
	.out(wire_sft14a_out[3:3]));
	soft   sft14a_4
	( 
	.in(wire_sft14a_in[4:4]),
	.out(wire_sft14a_out[4:4]));
	assign
		wire_sft14a_in = {w205w, {w205w, {w_sum_node10w[19], {w205w, w_sum_node10w[17]}}}};
	soft   sft15a_0
	( 
	.in(wire_sft15a_in[0:0]),
	.out(wire_sft15a_out[0:0]));
	soft   sft15a_1
	( 
	.in(wire_sft15a_in[1:1]),
	.out(wire_sft15a_out[1:1]));
	soft   sft15a_2
	( 
	.in(wire_sft15a_in[2:2]),
	.out(wire_sft15a_out[2:2]));
	soft   sft15a_3
	( 
	.in(wire_sft15a_in[3:3]),
	.out(wire_sft15a_out[3:3]));
	soft   sft15a_4
	( 
	.in(wire_sft15a_in[4:4]),
	.out(wire_sft15a_out[4:4]));
	assign
		wire_sft15a_in = {w205w, {w205w, {w205w, {2{w205w}}}}};
	soft   sft16a_0
	( 
	.in(wire_sft16a_in[0:0]),
	.out(wire_sft16a_out[0:0]));
	soft   sft16a_1
	( 
	.in(wire_sft16a_in[1:1]),
	.out(wire_sft16a_out[1:1]));
	soft   sft16a_2
	( 
	.in(wire_sft16a_in[2:2]),
	.out(wire_sft16a_out[2:2]));
	soft   sft16a_3
	( 
	.in(wire_sft16a_in[3:3]),
	.out(wire_sft16a_out[3:3]));
	soft   sft16a_4
	( 
	.in(wire_sft16a_in[4:4]),
	.out(wire_sft16a_out[4:4]));
	assign
		wire_sft16a_in = wire_add12_result;
	soft   sft3a_0
	( 
	.in(wire_sft3a_in[0:0]),
	.out(wire_sft3a_out[0:0]));
	soft   sft3a_1
	( 
	.in(wire_sft3a_in[1:1]),
	.out(wire_sft3a_out[1:1]));
	soft   sft3a_2
	( 
	.in(wire_sft3a_in[2:2]),
	.out(wire_sft3a_out[2:2]));
	soft   sft3a_3
	( 
	.in(wire_sft3a_in[3:3]),
	.out(wire_sft3a_out[3:3]));
	soft   sft3a_4
	( 
	.in(wire_sft3a_in[4:4]),
	.out(wire_sft3a_out[4:4]));
	soft   sft3a_5
	( 
	.in(wire_sft3a_in[5:5]),
	.out(wire_sft3a_out[5:5]));
	soft   sft3a_6
	( 
	.in(wire_sft3a_in[6:6]),
	.out(wire_sft3a_out[6:6]));
	soft   sft3a_7
	( 
	.in(wire_sft3a_in[7:7]),
	.out(wire_sft3a_out[7:7]));
	soft   sft3a_8
	( 
	.in(wire_sft3a_in[8:8]),
	.out(wire_sft3a_out[8:8]));
	soft   sft3a_9
	( 
	.in(wire_sft3a_in[9:9]),
	.out(wire_sft3a_out[9:9]));
	soft   sft3a_10
	( 
	.in(wire_sft3a_in[10:10]),
	.out(wire_sft3a_out[10:10]));
	soft   sft3a_11
	( 
	.in(wire_sft3a_in[11:11]),
	.out(wire_sft3a_out[11:11]));
	soft   sft3a_12
	( 
	.in(wire_sft3a_in[12:12]),
	.out(wire_sft3a_out[12:12]));
	soft   sft3a_13
	( 
	.in(wire_sft3a_in[13:13]),
	.out(wire_sft3a_out[13:13]));
	soft   sft3a_14
	( 
	.in(wire_sft3a_in[14:14]),
	.out(wire_sft3a_out[14:14]));
	soft   sft3a_15
	( 
	.in(wire_sft3a_in[15:15]),
	.out(wire_sft3a_out[15:15]));
	soft   sft3a_16
	( 
	.in(wire_sft3a_in[16:16]),
	.out(wire_sft3a_out[16:16]));
	assign
		wire_sft3a_in = {w192w, {wire_sft11a_out[11], {w192w, {w192w, {(~ w_lone_sum_node11w[6]), {w_sum_node10w[31], {w_sum_node10w[30], {w_sum_node10w[29], {(~ w_sum_node10w[10]), {w_sum_node10w[18], {w_sum_node10w[8], {w_sum_node10w[7], {w_sum_node10w[6], {w_sum_node10w[5], {w_sum_node10w[4], {w_sum_node10w[3:2]}}}}}}}}}}}}}}}};
	soft   sft4a_0
	( 
	.in(wire_sft4a_in[0:0]),
	.out(wire_sft4a_out[0:0]));
	soft   sft4a_1
	( 
	.in(wire_sft4a_in[1:1]),
	.out(wire_sft4a_out[1:1]));
	soft   sft4a_2
	( 
	.in(wire_sft4a_in[2:2]),
	.out(wire_sft4a_out[2:2]));
	soft   sft4a_3
	( 
	.in(wire_sft4a_in[3:3]),
	.out(wire_sft4a_out[3:3]));
	soft   sft4a_4
	( 
	.in(wire_sft4a_in[4:4]),
	.out(wire_sft4a_out[4:4]));
	soft   sft4a_5
	( 
	.in(wire_sft4a_in[5:5]),
	.out(wire_sft4a_out[5:5]));
	soft   sft4a_6
	( 
	.in(wire_sft4a_in[6:6]),
	.out(wire_sft4a_out[6:6]));
	soft   sft4a_7
	( 
	.in(wire_sft4a_in[7:7]),
	.out(wire_sft4a_out[7:7]));
	soft   sft4a_8
	( 
	.in(wire_sft4a_in[8:8]),
	.out(wire_sft4a_out[8:8]));
	soft   sft4a_9
	( 
	.in(wire_sft4a_in[9:9]),
	.out(wire_sft4a_out[9:9]));
	soft   sft4a_10
	( 
	.in(wire_sft4a_in[10:10]),
	.out(wire_sft4a_out[10:10]));
	soft   sft4a_11
	( 
	.in(wire_sft4a_in[11:11]),
	.out(wire_sft4a_out[11:11]));
	soft   sft4a_12
	( 
	.in(wire_sft4a_in[12:12]),
	.out(wire_sft4a_out[12:12]));
	soft   sft4a_13
	( 
	.in(wire_sft4a_in[13:13]),
	.out(wire_sft4a_out[13:13]));
	soft   sft4a_14
	( 
	.in(wire_sft4a_in[14:14]),
	.out(wire_sft4a_out[14:14]));
	soft   sft4a_15
	( 
	.in(wire_sft4a_in[15:15]),
	.out(wire_sft4a_out[15:15]));
	soft   sft4a_16
	( 
	.in(wire_sft4a_in[16:16]),
	.out(wire_sft4a_out[16:16]));
	assign
		wire_sft4a_in = {w205w, {w205w, {w_lone_sum_node11w[8], {(~ w_lone_sum_node11w[7]), {w_sum_node10w[41], {w_sum_node10w[40], {wire_sft11a_out[6], {wire_sft11a_out[5], {wire_sft11a_out[4], {wire_sft11a_out[3], {wire_sft11a_out[2], {w_sum_node10w[16], {w_sum_node10w[15], {w_sum_node10w[14], {w_sum_node10w[13], {w_sum_node10w[12:11]}}}}}}}}}}}}}}}};
	soft   sft5a_0
	( 
	.in(wire_sft5a_in[0:0]),
	.out(wire_sft5a_out[0:0]));
	soft   sft5a_1
	( 
	.in(wire_sft5a_in[1:1]),
	.out(wire_sft5a_out[1:1]));
	soft   sft5a_2
	( 
	.in(wire_sft5a_in[2:2]),
	.out(wire_sft5a_out[2:2]));
	soft   sft5a_3
	( 
	.in(wire_sft5a_in[3:3]),
	.out(wire_sft5a_out[3:3]));
	soft   sft5a_4
	( 
	.in(wire_sft5a_in[4:4]),
	.out(wire_sft5a_out[4:4]));
	soft   sft5a_5
	( 
	.in(wire_sft5a_in[5:5]),
	.out(wire_sft5a_out[5:5]));
	soft   sft5a_6
	( 
	.in(wire_sft5a_in[6:6]),
	.out(wire_sft5a_out[6:6]));
	soft   sft5a_7
	( 
	.in(wire_sft5a_in[7:7]),
	.out(wire_sft5a_out[7:7]));
	soft   sft5a_8
	( 
	.in(wire_sft5a_in[8:8]),
	.out(wire_sft5a_out[8:8]));
	soft   sft5a_9
	( 
	.in(wire_sft5a_in[9:9]),
	.out(wire_sft5a_out[9:9]));
	soft   sft5a_10
	( 
	.in(wire_sft5a_in[10:10]),
	.out(wire_sft5a_out[10:10]));
	soft   sft5a_11
	( 
	.in(wire_sft5a_in[11:11]),
	.out(wire_sft5a_out[11:11]));
	soft   sft5a_12
	( 
	.in(wire_sft5a_in[12:12]),
	.out(wire_sft5a_out[12:12]));
	soft   sft5a_13
	( 
	.in(wire_sft5a_in[13:13]),
	.out(wire_sft5a_out[13:13]));
	soft   sft5a_14
	( 
	.in(wire_sft5a_in[14:14]),
	.out(wire_sft5a_out[14:14]));
	soft   sft5a_15
	( 
	.in(wire_sft5a_in[15:15]),
	.out(wire_sft5a_out[15:15]));
	soft   sft5a_16
	( 
	.in(wire_sft5a_in[16:16]),
	.out(wire_sft5a_out[16:16]));
	assign
		wire_sft5a_in = {w205w, {w205w, {wire_sft11a_out[10], {wire_sft11a_out[9], {wire_sft11a_out[8], {wire_sft11a_out[7], {wire_sft16a_out[4], {wire_sft16a_out[3], {wire_sft16a_out[2], {wire_sft16a_out[1], {wire_sft16a_out[0], {wire_sft11a_out[1], {wire_sft11a_out[0], {w_sum_node10w[23], {w_sum_node10w[22], {2{w205w}}}}}}}}}}}}}}}}};
	soft   sft6a_0
	( 
	.in(wire_sft6a_in[0:0]),
	.out(wire_sft6a_out[0:0]));
	soft   sft6a_1
	( 
	.in(wire_sft6a_in[1:1]),
	.out(wire_sft6a_out[1:1]));
	soft   sft6a_2
	( 
	.in(wire_sft6a_in[2:2]),
	.out(wire_sft6a_out[2:2]));
	soft   sft6a_3
	( 
	.in(wire_sft6a_in[3:3]),
	.out(wire_sft6a_out[3:3]));
	soft   sft6a_4
	( 
	.in(wire_sft6a_in[4:4]),
	.out(wire_sft6a_out[4:4]));
	soft   sft6a_5
	( 
	.in(wire_sft6a_in[5:5]),
	.out(wire_sft6a_out[5:5]));
	soft   sft6a_6
	( 
	.in(wire_sft6a_in[6:6]),
	.out(wire_sft6a_out[6:6]));
	soft   sft6a_7
	( 
	.in(wire_sft6a_in[7:7]),
	.out(wire_sft6a_out[7:7]));
	soft   sft6a_8
	( 
	.in(wire_sft6a_in[8:8]),
	.out(wire_sft6a_out[8:8]));
	soft   sft6a_9
	( 
	.in(wire_sft6a_in[9:9]),
	.out(wire_sft6a_out[9:9]));
	soft   sft6a_10
	( 
	.in(wire_sft6a_in[10:10]),
	.out(wire_sft6a_out[10:10]));
	soft   sft6a_11
	( 
	.in(wire_sft6a_in[11:11]),
	.out(wire_sft6a_out[11:11]));
	soft   sft6a_12
	( 
	.in(wire_sft6a_in[12:12]),
	.out(wire_sft6a_out[12:12]));
	soft   sft6a_13
	( 
	.in(wire_sft6a_in[13:13]),
	.out(wire_sft6a_out[13:13]));
	soft   sft6a_14
	( 
	.in(wire_sft6a_in[14:14]),
	.out(wire_sft6a_out[14:14]));
	soft   sft6a_15
	( 
	.in(wire_sft6a_in[15:15]),
	.out(wire_sft6a_out[15:15]));
	soft   sft6a_16
	( 
	.in(wire_sft6a_in[16:16]),
	.out(wire_sft6a_out[16:16]));
	assign
		wire_sft6a_in = wire_add2_result;
	soft   sft8a_0
	( 
	.in(wire_sft8a_in[0:0]),
	.out(wire_sft8a_out[0:0]));
	soft   sft8a_1
	( 
	.in(wire_sft8a_in[1:1]),
	.out(wire_sft8a_out[1:1]));
	soft   sft8a_2
	( 
	.in(wire_sft8a_in[2:2]),
	.out(wire_sft8a_out[2:2]));
	soft   sft8a_3
	( 
	.in(wire_sft8a_in[3:3]),
	.out(wire_sft8a_out[3:3]));
	soft   sft8a_4
	( 
	.in(wire_sft8a_in[4:4]),
	.out(wire_sft8a_out[4:4]));
	soft   sft8a_5
	( 
	.in(wire_sft8a_in[5:5]),
	.out(wire_sft8a_out[5:5]));
	soft   sft8a_6
	( 
	.in(wire_sft8a_in[6:6]),
	.out(wire_sft8a_out[6:6]));
	soft   sft8a_7
	( 
	.in(wire_sft8a_in[7:7]),
	.out(wire_sft8a_out[7:7]));
	soft   sft8a_8
	( 
	.in(wire_sft8a_in[8:8]),
	.out(wire_sft8a_out[8:8]));
	soft   sft8a_9
	( 
	.in(wire_sft8a_in[9:9]),
	.out(wire_sft8a_out[9:9]));
	soft   sft8a_10
	( 
	.in(wire_sft8a_in[10:10]),
	.out(wire_sft8a_out[10:10]));
	soft   sft8a_11
	( 
	.in(wire_sft8a_in[11:11]),
	.out(wire_sft8a_out[11:11]));
	assign
		wire_sft8a_in = {w205w, {(~ w_sum_node10w[43]), {w_sum_node10w[42], {(~ w_sum_node10w[32]), {w192w, {(~ w_lone_sum_node11w[4]), {(~ w_lone_sum_node11w[3]), {w_sum_node10w[37], {w_sum_node10w[36], {w_sum_node10w[35], {w_sum_node10w[34:33]}}}}}}}}}}};
	soft   sft9a_0
	( 
	.in(wire_sft9a_in[0:0]),
	.out(wire_sft9a_out[0:0]));
	soft   sft9a_1
	( 
	.in(wire_sft9a_in[1:1]),
	.out(wire_sft9a_out[1:1]));
	soft   sft9a_2
	( 
	.in(wire_sft9a_in[2:2]),
	.out(wire_sft9a_out[2:2]));
	soft   sft9a_3
	( 
	.in(wire_sft9a_in[3:3]),
	.out(wire_sft9a_out[3:3]));
	soft   sft9a_4
	( 
	.in(wire_sft9a_in[4:4]),
	.out(wire_sft9a_out[4:4]));
	soft   sft9a_5
	( 
	.in(wire_sft9a_in[5:5]),
	.out(wire_sft9a_out[5:5]));
	soft   sft9a_6
	( 
	.in(wire_sft9a_in[6:6]),
	.out(wire_sft9a_out[6:6]));
	soft   sft9a_7
	( 
	.in(wire_sft9a_in[7:7]),
	.out(wire_sft9a_out[7:7]));
	soft   sft9a_8
	( 
	.in(wire_sft9a_in[8:8]),
	.out(wire_sft9a_out[8:8]));
	soft   sft9a_9
	( 
	.in(wire_sft9a_in[9:9]),
	.out(wire_sft9a_out[9:9]));
	soft   sft9a_10
	( 
	.in(wire_sft9a_in[10:10]),
	.out(wire_sft9a_out[10:10]));
	soft   sft9a_11
	( 
	.in(wire_sft9a_in[11:11]),
	.out(wire_sft9a_out[11:11]));
	assign
		wire_sft9a_in = {w205w, {w205w, {w205w, {w205w, {(~ w_lone_sum_node11w[5]), {w_sum_node10w[39], {w_sum_node10w[38], {w_sum_node10w[28], {w_sum_node10w[27], {w_sum_node10w[26], {w_sum_node10w[25:24]}}}}}}}}}}};
	assign
		dataa_node = {dataa[8:0]},
		datab_node = {datab[8:0]},
		final_result_node = {w339w[17:0]},
		result = {final_result_node[17:0]},
		w192w = 1'b1,
		w205w = 1'b0,
		w339w = {wire_sft6a_out[16:15], wire_sft6a_out[14:13], wire_sft6a_out[12:11], wire_sft6a_out[10:9], wire_sft6a_out[8:7], wire_sft6a_out[6:5], wire_sft6a_out[4:3], wire_sft6a_out[2:1], wire_sft6a_out[0], w_sum_node10w[1], w_sum_node10w[0]},
		w_decoder_node9w = {({dataa_node[8:0]} & {9{w_le_datab_node8w[8]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[7]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[6]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[5]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[4]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[3]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[2]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[1]}}), ({dataa_node[8:0]} & {9{w_le_datab_node8w[0]}})},
		w_le_datab_node8w = {datab_node[8:0]},
		w_lone_sum_node11w = {w_decoder_node9w[80:72]},
		w_sum_node10w = {wire_sum_adder1aa_3result[9:0], w_decoder_node9w[54], wire_sum_adder1aa_2result[9:0], w_decoder_node9w[36], wire_sum_adder1aa_1result[9:0], w_decoder_node9w[18], wire_sum_adder1aa_0result[9:0], w_decoder_node9w[0]};
endmodule //mult_7a11
//VALID FILE
