export SGL_USE_CUTLASS_MOE_FP8=1 
export CUTLASS_TUNE_SWITCH=1 
export USE_STATIC_SCALE=1
export USE_FAST_TOPK_WEIGHTS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

NCCL_MIN_NCHANNELS=24 NCCL_IB_QPS_PER_CONNECTION=8 SGLANG_USE_MODELSCOPE=1 \
 python3 -m sglang.launch_server --model-path /root/models/Qwen3-235B-A22B-Thinking-2507-FP8 \
  --tensor-parallel-size 4 --reasoning-parser qwen3 \
  --cuda-graph-max-bs 96 --cuda-graph-bs 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 34 38 42 46 50 56 64 70 78 80 88 96 \
  --max-running-requests 96 --attention-backend fa3 --mem-fraction-static 0.8 --disable-radix-cache --kv-cache-dtype fp8_e4m3 \
  --speculative-algo EAGLE3 --speculative-draft /root/models/Qwen3-235B-A22B-Thinking-2507-FP8-eagle3-0828 --speculative-num-steps 2 \
  --speculative-eagle-topk 2 --speculative-num-draft-tokens 7 