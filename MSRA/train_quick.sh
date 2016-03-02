#!/usr/bin/env sh

TOOLS=./build/tools
$TOOLS/caffe train  --solver=MSRA/msra_solver.prototxt

