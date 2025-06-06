#!/bin/bash
set -x
set -e

# source activate py37
PHASE=$1
GPUS=$2
DATA=$3
BATCH_SIZE=$4
DATASET_PATH=$5

LOG_DIR="./logs"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

LOG="${LOG_DIR}/${PHASE}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

OUT_DIR=output/UString/vgg16

# experiments on DAD dataset
case ${PHASE} in
  train)
    CUDA_VISIBLE_DEVICES=$GPUS python main.py \
      --data_path $DATASET_PATH/obj_feat \
      --dataset $DATA \
      --feature_name vgg16 \
      --phase train \
      --base_lr 0.0005 \
      --batch_size $BATCH_SIZE \
      --gpus $GPUS \
      --output_dir $OUT_DIR
    ;;
  test)
    CUDA_VISIBLE_DEVICES=$GPUS python main.py \
      --data_path $DATASET_PATH/obj_feat \
      --dataset $DATA \
      --feature_name vgg16 \
      --phase test \
      --batch_size $BATCH_SIZE \
      --gpus $GPUS \
      --visualize \
      --output_dir $OUT_DIR \
      --model_file $OUT_DIR/$DATA/snapshot/final_model.pth
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac

