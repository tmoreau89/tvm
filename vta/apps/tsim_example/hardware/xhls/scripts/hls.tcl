#
#  Copyright (c) 2018 by Contributors
#  file: hls.tcl
#  brief: HLS generation script.
#

open_project vadd
set_top vadd
add_files src/vadd.cc
add_files -tb src/vadd_test.cc
open_solution "solution0"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csim_design -clean
csynth_design
close_project

exit
