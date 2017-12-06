#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="ctw1500"
ITERS=78000


LOG="experiments/logs/rfcn_ctd_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/ctd/solver_ctd.prototxt \
  --weights data/imagenet_models/ResNet-50-model.caffemodel \
  --iters ${ITERS} \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/rfcn_ctd.yml \
  --trainval_label data/ctw1500/train/trainval_label_curve.txt \
  --trainval_image data/ctw1500/train/trainval.txt \
  ${EXTRA_ARGS} 

 
