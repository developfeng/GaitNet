#!/usr/bin/env sh
CAFFE=YOUR_CAFFE_ROTE #Example:CAFFE=/home/user1/caffe/build/tools/caffe

#SegNet
LOG=./gaitnet-seg-`date +%Y-%m-%d-%H-%M-%S`.log
$CAFFE train --solver=./solver-seg.prototxt --gpu=0 2>&1 | tee $LOG

#RecNet
#LOG=./gaitnet-rec-`date +%Y-%m-%d-%H-%M-%S`.log
#$CAFFE train --solver=./solver-rec.prototxt --weights=./gaitnet-seg_iter_70000.caffemodel --gpu=0 2>&1 | tee $LOG

#Joint
#LOG=./gaitnet-joint-`date +%Y-%m-%d-%H-%M-%S`.log
#$CAFFE train --solver=./solver-joint.prototxt --weights=./gaitnet-rec_iter_60000.caffemodel --gpu=0 2>&1 | tee $LOG

#RecNet-Sia
#LOG=./gaitnet-sia-`date +%Y-%m-%d-%H-%M-%S`.log
#$CAFFE train --solver=./solver-sia.prototxt --weights=./gaitnet-joint_iter_30000.caffemodel --gpu=0 2>&1 | tee $LOG
