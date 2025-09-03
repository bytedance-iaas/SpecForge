
# 原始的调用正常训练的脚本
# export NUM_GPUS=8
# torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     scripts/train_eagle3_offline.py \
#     --target-model-path /root/models/Qwen3-235B-A22B-Thinking-2507-FP8 \
#     --draft-model-config ./configs/qwen3-235B-A22B-eagle3.json \
#     --train-data-path ./cache/dataset/train.jsonl \
#     --train-hidden-states-path ./cache/hidden_states/ \
#     --output-dir ./outputs/Qwen3-235B-A22B-Thinking-2507-FP8-eagle3 \
#     --num-epochs 10 \
#     --draft-global-batch-size 16 \
#     --draft-micro-batch-size 1 \
#     --learning-rate 5e-5 \
#     --draft-attention-backend flex_attention \
#     --max-length 2048 \
#     --chat-template qwen \
#     --cache-dir ./cache \
#     --dist-timeout=10 \
#     --log-steps 1 \


# 更新后可以中断重启的训练脚本
#!/bin/bash

# Qwen3-235B Eagle3 Training Script with Resume Support

set -e  # Exit on any error

# Configuration
export NUM_GPUS=8
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Default values
RESUME_MODE="none"  # none, auto, epoch
RESUME_EPOCH=""
STRICT_LOAD="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_MODE="auto"
            shift
            ;;
        --resume-epoch)
            RESUME_MODE="epoch"
            RESUME_EPOCH="$2"
            shift 2
            ;;
        --strict-load)
            STRICT_LOAD="true"
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "=== Qwen3-235B Eagle3 Training ==="
echo "GPUs: $NUM_GPUS"
echo "Resume mode: $RESUME_MODE"
if [[ "$RESUME_MODE" == "epoch" ]]; then
    echo "Resume from epoch: $RESUME_EPOCH"
fi
echo "Strict load: $STRICT_LOAD"
echo "=================================="

# Build command
CMD="torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    ./scripts/retrain_eagle3_offline.py \
    --target-model-path /root/models/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --draft-model-config ./configs/qwen3-235B-A22B-eagle3.json \
    --train-data-path ./cache/dataset/train.jsonl \
    --train-hidden-states-path ./cache/hidden_states/ \
    --output-dir ./outputs/Qwen3-235B-A22B-Thinking-2507-FP8-eagle3 \
    --num-epochs 10 \
    --draft-global-batch-size 16 \
    --draft-micro-batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir ./cache \
    --dist-timeout=10 \
    --log-steps 1 \
    --warmup-ratio 0.015 \
    --max-grad-norm 0.5 \
    --ttt-length 7 \
    --save-interval 1 \
    --eval-interval 1 \
    --report-to none \
    --build-dataset-num-proc 8"

# Add resume arguments based on mode
if [[ "$RESUME_MODE" == "auto" ]]; then
    CMD="$CMD --resume"
elif [[ "$RESUME_MODE" == "epoch" ]]; then
    CMD="$CMD --resume-epoch $RESUME_EPOCH"
fi

# Add strict load if specified
if [[ "$STRICT_LOAD" == "true" ]]; then
    CMD="$CMD --strict-load"
fi

# Execute command
echo "Executing: $CMD"
eval $CMD

echo "=== Training completed ==="