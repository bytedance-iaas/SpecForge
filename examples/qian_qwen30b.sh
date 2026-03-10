cd /data01/qian_dev/SpecForge

CUDA_VISIBLE_DEVICES=6 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
TORCHINDUCTOR_CACHE_DIR=/data01/qian_dev/SpecForge/cache/compiled_kernels \
BUILD_DATASET_NUM_PROC=1 \
HF_DATASETS_DISABLE_CACHING=1 \
USE_ENGRAM=1 \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node 1 \
  /data01/qian_dev/SpecForge/scripts/train_eagle3.py \
  --target-model-path /data00/models/Qwen3-30B-A3B \
  --draft-model-config /data01/qian_dev/SpecForge/configs/qwen3-30B-A3B-eagle3.json \
  --train-data-path /data01/qian_dev/datasets/sharegpt_train_converted_all.jsonl \
  --build-dataset-num-proc 1 \
  --output-dir /data01/qian_dev/SpecForge/outputs/qwen3-30b-a3b-instruct-eagle3-sharegpt-engram \
  --num-epochs 2 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --max-length 4096 \
  --chat-template qwen \
  --cache-dir /data01/qian_dev/SpecForge/cache \
  --embedding-key model.embed_tokens.weight \
  --tp-size 1 \
  --target-model-backend sglang \
  --sglang-mem-fraction-static 0.8