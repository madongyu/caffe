#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe test --model=MSRA/msra.prototxt --weights=MSRA/msra_iter_4613.caffemodel
