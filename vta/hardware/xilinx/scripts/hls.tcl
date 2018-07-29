#
#  Copyright (c) 2018 by Contributors
#  file: hls.tcl
#  brief: HLS generation script.
#

# Command line arguments:
# Arg 1: path to design sources
# Arg 2: path to sim sources
# Arg 3: path to test sources
# Arg 4: path to include sources
# Arg 5: mode
# Arg 6: debug
# Arg 7: no_dsp
# Arg 8: no_alu
# Arg 9: target clock period
# Arg 10: target II for GEMM
# Arg 11: target II for tensor ALU
# Arg 12: input type width (log)
# Arg 13: weight type width (log)
# Arg 14: accum type width (log)
# Arg 15: output type width (log)
# Arg 16: batch size (log)
# Arg 17: in block size (log)
# Arg 18: out block size (log)
# Arg 19: uop buffer size in B (log)
# Arg 20: inp buffer size in B (log)
# Arg 21: wgt buffer size in B (log)
# Arg 22: acc buffer size in B (log)
# Arg 23: out buffer size in B (log)

if { [llength $argv] eq 25 } {
	set src_dir [lindex $argv 2]
	set sim_dir [lindex $argv 3]
	set test_dir [lindex $argv 4]
	set include_dir [lindex $argv 5]
	set mode [lindex $argv 6]
	set debug [lindex $argv 7]
	set no_dsp [lindex $argv 8]
	set no_alu [lindex $argv 9]
	set target_period [lindex $argv 10]
	set target_gemm_ii [lindex $argv 11]
	set target_alu_ii [lindex $argv 12]
	set inp_width [lindex $argv 13]
	set wgt_width [lindex $argv 14]
	set acc_width [lindex $argv 15]
	set out_width [lindex $argv 16]
	set batch [lindex $argv 17]
	set block_in [lindex $argv 18]
	set block_out [lindex $argv 19]
	set uop_buff_size [lindex $argv 20]
	set inp_buff_size [lindex $argv 21]
	set wgt_buff_size [lindex $argv 22]
	set acc_buff_size [lindex $argv 23]
	set out_buff_size [lindex $argv 24]
} else {
	set src_dir "../src"
	set sim_dir "../sim"
	set test_dir "../../src/test"
	set include_dir "../../include"
	set mode "all"
	set debug "false"
	set no_dsp "true"
	set no_alu "false"
	set target_period 8
	set target_gemm_ii 10
	set target_alu_ii 16
	set inp_width 3
	set wgt_width 3
	set acc_width 5
	set out_width 3
	set batch 1
	set block_in 4
	set block_out 4
	set uop_buff_size 15
	set inp_buff_size 15
	set wgt_buff_size 15
	set acc_buff_size 17
	set out_buff_size 15
	exit
}

# Initializes the HLS design and sets HLS pragmas for memory partitioning.
# This is necessary because of a Vivado restriction that doesn't allow for
# buses wider than 1024 bits.
proc init_design {per ii inp_width wgt_width out_width acc_width batch block_in block_out no_alu} {

	# Set device number
	set_part {xc7z020clg484-1}

	# Max bus width (supported by Vivado)
	set max_width 1024

	# Set axi width (TODO derive from top level config)
	set axi_width 64

	# Set the clock frequency
	create_clock -period $per -name default

	# Set pipeline directive
	set_directive_pipeline -II $ii "compute/READ_GEMM_UOP"

	if {$no_alu=="false"} {
		set_directive_pipeline -II $ii "compute/READ_ALU_UOP"
	}

	# Set input partition factor to (INP_VECTOR_WIDTH*BATCH/(1024*ii)
	set inp_bus_width [expr {(1 << ($inp_width + $block_in + $batch)) / $ii}]
	set inp_partition_factor [expr {$inp_bus_width / $max_width}]
	if {$inp_partition_factor == 0} {
		set inp_reshape_factor [expr {$inp_bus_width / $axi_width}]
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "load" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "compute" inp_mem
	} else {
		set inp_reshape_factor [expr {$max_width / $axi_width}]
		set_directive_array_partition -type block -factor $inp_partition_factor -dim 2 "load" inp_mem
		set_directive_array_partition -type block -factor $inp_partition_factor -dim 2 "compute" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "load" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "compute" inp_mem
	}
	# Set weight partition factor to (WGT_VECTOR_WIDTH*BLOCK_OUT/(1024*ii))
	set wgt_bus_width [expr {(1 << ($wgt_width + $block_in + $block_out)) / $ii}]
	set wgt_partition_factor [expr {$wgt_bus_width / $max_width}]
	if {$wgt_partition_factor == 0} {
		set wgt_reshape_factor [expr {$wgt_bus_width / $axi_width}]
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "load" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "compute" wgt_mem
	} else {
		set wgt_reshape_factor [expr {$max_width / $axi_width}]
		set_directive_array_partition -type block -factor $wgt_partition_factor -dim 2 "load" wgt_mem
		set_directive_array_partition -type block -factor $wgt_partition_factor -dim 2 "compute" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "load" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "compute" wgt_mem
	}
	# Set output partition factor to (OUT_VECTOR_WIDTH*BATCH/(1024*ii))
	set out_bus_width [expr {(1 << ($out_width + $block_out + $batch)) / $ii}]
	set out_partition_factor [expr {$out_bus_width / $max_width}]
	if {$out_partition_factor == 0} {
		set out_reshape_factor [expr {$out_bus_width / $axi_width}]
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "compute" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "store" out_mem
	} else {
		set out_reshape_factor [expr {$max_width / $axi_width}]
		set_directive_array_partition -type block -factor $out_partition_factor -dim 2 "compute" out_mem
		set_directive_array_partition -type block -factor $out_partition_factor -dim 2 "store" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "compute" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "store" out_mem
	}
	# Set accumulator partition factor
	# set acc_bus_width [expr {(1 << ($acc_width + $block_out + $batch)) / $ii}]
	# set acc_reshape_factor [expr {$acc_bus_width / $axi_width}]
	# set_directive_array_reshape -type block -factor $acc_reshape_factor -dim 2 "compute" acc_mem
}

# C define flags to pass to compiler
set cflags "-I $include_dir -I $src_dir -I $test_dir \
	-DVTA_LOG_WGT_WIDTH=$wgt_width -DVTA_LOG_INP_WIDTH=$inp_width \
	-DVTA_LOG_ACC_WIDTH=$acc_width -DVTA_LOG_OUT_WIDTH=$out_width \
	-DVTA_LOG_BATCH=$batch -DVTA_LOG_BLOCK_OUT=$block_out -DVTA_LOG_BLOCK_IN=$block_in \
	-DVTA_LOG_UOP_BUFF_SIZE=$uop_buff_size -DVTA_LOG_INP_BUFF_SIZE=$inp_buff_size \
	-DVTA_LOG_WGT_BUFF_SIZE=$wgt_buff_size -DVTA_LOG_ACC_BUFF_SIZE=$acc_buff_size \
	-DVTA_LOG_OUT_BUFF_SIZE=$out_buff_size"
if {$debug=="true"} {
	append cflags " -DVTA_DEBUG=1"
}
if {$no_dsp=="true"} {
	append cflags " -DNO_DSP"
}
if {$no_alu=="true"} {
	append cflags " -DNO_ALU"
}

# HLS behavioral sim
if {$mode=="all" || $mode=="sim"} {
	open_project vta_sim
	set_top vta
	add_files $src_dir/vta.cc -cflags $cflags
	add_files -tb $sim_dir/vta_test.cc -cflags $cflags
	add_files -tb $test_dir/test_lib.cc -cflags $cflags
	open_solution "solution0"
	init_design $target_period $target_gemm_ii $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $no_alu
	csim_design -clean
	close_project
}

# Generate fetch stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="fetch"} {
	open_project vta_fetch
	set_top fetch
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target_period $target_gemm_ii $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $no_alu
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

# Generate load stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="load"} {
	open_project vta_load
	set_top load
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target_period $target_gemm_ii $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $no_alu
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

# Generate compute stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="compute"} {
	open_project vta_compute
	set_top compute
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target_period $target_gemm_ii $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $no_alu
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

# Generate store stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="store"} {
	open_project vta_store
	set_top store
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target_period $target_gemm_ii $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $no_alu
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

exit

