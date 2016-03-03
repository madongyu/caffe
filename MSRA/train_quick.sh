#!/usr/bin/env sh

TOOLS=./build/tools
GLOG_logtostderr=0 GLOG_log_dir=../mdylog  $TOOLS/caffe train  --solver=MSRA/msra_solver.prototxt

