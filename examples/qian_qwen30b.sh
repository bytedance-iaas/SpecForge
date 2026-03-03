#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=1
TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-1}

torchrun \
    --standalone \
    --nproc_per_node 1 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /data02/models/Qwen3-30B-A3B \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
    --train-data-path /data01/qian_dev/datasets/sharegpt_train_converted.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-30b-a3b-instruct-eagle3-sharegpt \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
