#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import Dataset
from specforge import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
)
from specforge.args import SGLangBackendArgs
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    get_eagle3_target_model,
)
from specforge.utils import (
    create_draft_config_from_target,
    print_args_with_dots,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
    safe_conversations_generator,
)


# ----------------------------
# Args
# ----------------------------
def parse_args() -> Tuple[ArgumentParser, Namespace]:
    parser = argparse.ArgumentParser(description="Evaluate an EAGLE3 draft model (non-VLM)")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help="Draft config path. If omitted, auto-generated from target model.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key of embedding weight in target model",
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="Backend used to run the target model online",
    )
    model_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the *target* model",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--eval-data-path", type=str, required=True)
    data_group.add_argument("--chat-template", type=str, default="llama3")
    data_group.add_argument("--max-length", type=int, default=2048)
    data_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Input data is preformatted text with chat template already applied.",
    )
    data_group.add_argument(
        "--train-only-last-turn",
        action="store_true",
        help="Only last assistant turn contributes to loss mask.",
    )
    data_group.add_argument("--build-dataset-num-proc", type=int, default=8)
    data_group.add_argument("--dataloader-num-workers", type=int, default=4)

    eval_group = parser.add_argument_group("eval")
    eval_group.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Checkpoint dir produced by training, e.g. output/epoch_1_step_5000/",
    )
    eval_group.add_argument("--batch-size", type=int, default=1)
    eval_group.add_argument("--ttt-length", type=int, default=7)
    eval_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        help="Attention backend for draft model",
    )
    eval_group.add_argument("--seed", type=int, default=0)
    eval_group.add_argument("--dist-timeout", type=int, default=20)

    cache_group = parser.add_argument_group("cache")
    cache_group.add_argument("--cache-dir", type=str, default="./cache")

    other_group = parser.add_argument_group("others")
    other_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="Directory to download/cache the target model",
    )
    other_group.add_argument("--verbose", action="store_true")

    # sglang backend args
    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    args = parser.parse_args()

    # Non-VLM only: hard guard.
    # If you want, you can expose an --is-vlm flag and error out if set;
    # here we simply don't implement it.
    return parser, args


# ----------------------------
# Helpers
# ----------------------------
def sanity_check(args: Namespace) -> None:
    # dp_size = world_size / tp_size (target tp)
    world = dist.get_world_size()
    assert world % args.tp_size == 0, f"world_size={world} must be divisible by tp_size={args.tp_size}"
    args.dp_size = world // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


@torch.no_grad()
def run_forward_online(
    eagle3_model: torch.nn.Module,
    data: dict,
    target_model: Eagle3TargetModel,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Generate eagle3 supervision online via target model
    eagle3_data = target_model.generate_eagle3_data(
        input_ids=data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
        loss_mask=data["loss_mask"].cuda(),
    )

    input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
    attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
    loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
    target = get_dp_data_shard_from_tp(eagle3_data.target)
    hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)

    plosses, _, acces = eagle3_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        target=target,
        hidden_states=hidden_states,
        position_ids=(data["position_ids"].cuda() if "position_ids" in data else None),
        is_vlm=False,
    )
    return plosses, acces


def build_target_model(args: Namespace, draft_model_config: AutoDraftModelConfig) -> Eagle3TargetModel:
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    else:
        target_model_kwargs = {}

    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.model_download_dir,
        **target_model_kwargs,
        trust_remote_code=args.trust_remote_code,
    )

    # aux hidden state layers
    if (
        hasattr(draft_model_config, "eagle_config")
        and draft_model_config.eagle_config is not None
        and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
    ):
        target_model.set_aux_hidden_states_layers(
            draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
        )
    else:
        target_model.set_aux_hidden_states_layers()

    return target_model


def build_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, torch.nn.Module]:
    if args.draft_model_config is None:
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path,
            cache_dir=args.model_download_dir,
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    if not os.path.isdir(args.ckpt_dir):
        raise ValueError(f"--ckpt-dir must be a directory, got: {args.ckpt_dir}")

    # Load draft model weights from checkpoint dir
    draft_model = AutoEagle3DraftModel.from_pretrained(
        args.ckpt_dir,
        attention_backend=args.attention_backend,
        torch_dtype=torch.bfloat16,
    ).cuda()

    # Ensure embedding is aligned with target (same behavior as training)
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    return draft_model_config, draft_model


def build_eval_dataloader(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
) -> Tuple[DataLoader, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )

    cache_params_string = (
        f"{args.eval_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    eval_dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": args.eval_data_path},
    )

    with rank_0_priority():
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=False,
            is_preformatted=args.is_preformatted,
            processor=None,
            num_proc=args.build_dataset_num_proc,
            train_only_last_turn=args.train_only_last_turn,
        )

        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=eval_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )

    eval_dataloader = prepare_dp_dataloaders(
        eval_eagle3_dataset,
        args.target_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        process_group=get_dp_group(),
        is_vlm=False,
    )
    return eval_dataloader, vocab_mapping_path


def reduce_mean_tensor(x: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x


# ----------------------------
# Main
# ----------------------------
def main():
    parser, args = parse_args()
    set_seed(args.seed)

    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=1,
        sp_ulysses_size=1,
    )
    sanity_check(args)
    print_args_with_dots(args)
    print_with_rank("Initialized distributed environment (eval)")

    # Build models
    draft_model_config, draft_model = build_draft_model(args)
    target_model = build_target_model(args, draft_model_config)

    # Build dataloader + vocab mapping
    eval_dataloader, vocab_mapping_path = build_eval_dataloader(args, draft_model_config)
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # Build eagle3 wrapper (online eval)
    eagle3_model = OnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    ).cuda()
    eagle3_model.eval()

    # Accumulators (per ttt position)
    # We'll accumulate sums and counts to avoid storing everything.
    t = args.ttt_length
    sum_acc = torch.zeros(t, device="cuda", dtype=torch.float32)
    sum_ploss = torch.zeros(t, device="cuda", dtype=torch.float32)
    n_batches = torch.zeros(1, device="cuda", dtype=torch.float32)

    if dist.get_rank() == 0:
        pbar = tqdm(eval_dataloader, desc="Evaluating", leave=True)
    else:
        pbar = eval_dataloader

    with torch.no_grad():
        for data in pbar:
            plosses, acces = run_forward_online(eagle3_model, data, target_model)

            acc_t = torch.stack([a.float() for a in acces]).to("cuda")
            pl_t = torch.stack([p.float() for p in plosses]).to("cuda")

            # local sums
            sum_acc += acc_t
            sum_ploss += pl_t
            n_batches += 1.0

    # Reduce across all ranks (DP+TP world)
    sum_acc = reduce_mean_tensor(sum_acc)
    sum_ploss = reduce_mean_tensor(sum_ploss)
    n_batches = reduce_mean_tensor(n_batches)

    # Convert to means
    mean_acc = (sum_acc / n_batches).detach().cpu().tolist()
    mean_ploss = (sum_ploss / n_batches).detach().cpu().tolist()

    if dist.get_rank() == 0:
        print("\n=== EAGLE3 Draft Eval Results (non-VLM) ===")
        for i in range(t):
            print(f"pos {i:02d}:  acc={mean_acc[i]:.6f}   pLoss={mean_ploss[i]:.6f}")
        print("------------------------------------------")
        print(f"avg acc : {sum(mean_acc)/len(mean_acc):.6f}")
        print(f"avg pLoss: {sum(mean_ploss)/len(mean_ploss):.6f}")
        print(f"ckpt-dir: {args.ckpt_dir}")
        print(f"eval-data: {args.eval_data_path}")

    destroy_distributed()


if __name__ == "__main__":
    main()