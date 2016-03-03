#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=../mdylog $TOOLS/caffe test --model=MSRA/msra.prototxt --weights=MSRA/msra_iter_4613.caffemodel
