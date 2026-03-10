import argparse
import hashlib
import json
import math
import os
from argparse import Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from specforge import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_draft_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    TargetHead,
    get_eagle3_target_model,
)
from specforge.utils import (
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
    safe_conversations_generator,
)


def parse_args() -> Tuple[argparse.ArgumentParser, Namespace]:
    parser = argparse.ArgumentParser("Eval EAGLE3 accuracy without SGLang")

    # IO
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Draft model checkpoint dir to load.")
    parser.add_argument("--draft-model-config", type=str, default=None, help="Path to draft model config.json if no ckpt.")

    # Target & dataset
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--target-model-backend", type=str, default="hf", choices=["hf", "custom"], help="Use HF backend (online) or custom TargetHead (offline).")
    parser.add_argument("--model-download-dir", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    # Eval data (mutually exclusive)
    parser.add_argument("--eval-data-path", type=str, default=None, help="Path to eval conversations jsonl (online eval).")
    parser.add_argument("--eval-hidden-states-path", type=str, default=None, help="Path to precomputed hidden states (offline eval).")

    # Formatting & lengths
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--ttt-length", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--build-dataset-num-proc", type=int, default=4)

    # Attention backend
    parser.add_argument("--attention-backend", type=str, default="sdpa", choices=["sdpa", "fa", "usp", "flex_attention"]) 

    # Embedding / lm head keys
    parser.add_argument("--embedding-key", type=str, default="model.embed_tokens.weight")
    parser.add_argument("--lm-head-key", type=str, default="model.lm_head.weight")

    # VLM
    parser.add_argument("--is-vlm", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=802816)

    # Distributed
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dist-timeout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if (args.eval_data_path is None) == (args.eval_hidden_states_path is None):
        parser.error("You must set exactly one of --eval-data-path or --eval-hidden-states-path.")

    return parser, args


def build_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, torch.nn.Module]:
    if args.ckpt_dir is not None:
        # Load config from checkpoint dir
        config_path = os.path.join(args.ckpt_dir, "config.json")
        draft_model_config = AutoDraftModelConfig.from_file(config_path)
        draft_model = AutoEagle3DraftModel.from_pretrained(
            args.ckpt_dir, attention_backend=args.attention_backend, torch_dtype=torch.bfloat16
        ).cuda()
    else:
        if args.draft_model_config is None:
            raise ValueError("When --ckpt-dir is not provided, you must specify --draft-model-config.")
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config, attention_backend=args.attention_backend, torch_dtype=torch.bfloat16
        ).cuda()

    # Load and freeze embedding from target model
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    return draft_model_config, draft_model


def build_target_model(args: Namespace, draft_model_config: AutoDraftModelConfig, is_online: bool):
    if is_online:
            target_model = get_eagle3_target_model(
                pretrained_model_name_or_path=args.target_model_path,
                backend="hf",
                torch_dtype=torch.bfloat16,
                device="cuda",
                cache_dir=args.model_download_dir,
                trust_remote_code=args.trust_remote_code,
            )
            processor = None
        return target_model, processor
    else:
        # Offline: use TargetHead only (no forward through target model)
        target_head = TargetHead.from_pretrained(
            model_path=args.target_model_path,
            lm_head_key=args.lm_head_key,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
        )
        return target_head, None


def build_eval_dataloader(
    args: Namespace, draft_model_config: AutoDraftModelConfig, processor: Optional[AutoProcessor]
):
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=args.trust_remote_code)

    is_online = args.eval_data_path is not None and args.eval_hidden_states_path is None
    # Build dataset for eval and vocab mapping
    with rank_0_priority():
        if is_online:
            eval_dataset = Dataset.from_generator(
                generator=safe_conversations_generator, gen_kwargs={"file_path": args.eval_data_path}
            )
            eval_eagle3_dataset = build_eagle3_dataset(
                dataset=eval_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                cache_dir=os.path.join(args.output_dir, "cache", "processed_dataset"),
                cache_key=hashlib.md5(
                    f"{args.eval_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}".encode()
                ).hexdigest(),
                is_vlm=args.is_vlm,
                is_preformatted=False,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
                train_only_last_turn=False,
            )
        else:
            eval_eagle3_dataset = build_offline_eagle3_dataset(
                args.eval_hidden_states_path,
                args.max_length,
                ttt_length=args.ttt_length,
                use_usp_preprocess=(args.attention_backend == "usp"),
            )

        # Generate vocab mapping from eval dataset
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=eval_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.output_dir, "cache", "vocab_mapping"),
            cache_key=hashlib.md5(
                f"{args.target_model_path}-{args.max_length}-{args.chat_template}".encode()
            ).hexdigest(),
        )

    # Build dataloader
    eval_dataloader = prepare_dp_dataloaders(
        eval_eagle3_dataset,
        args.tp_size * args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        process_group=(get_draft_dp_group() if args.attention_backend == "usp" and not is_online else get_dp_group()),
        is_vlm=args.is_vlm,
    )

    return eval_dataloader, vocab_mapping_path, is_online


def run_forward(
    args: Namespace,
    eagle3_model: torch.nn.Module,
    data: dict,
    target_model: Optional[Eagle3TargetModel] = None,
    is_online: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # VLM path (online, custom backend)
    if args.is_vlm and is_online and hasattr(target_model, "generate_eagle3_data") is False:
        # Using Qwen2.5-VL direct HF model path
        plosses, _, acces = eagle3_model(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
            pixel_values=data.get("pixel_values", None).cuda() if data.get("pixel_values", None) is not None else None,
            image_grid_thw=data.get("image_grid_thw", None).cuda() if data.get("image_grid_thw", None) is not None else None,
        )
        return plosses, acces

    # Text path
    image_grid_thw = None
    if is_online:
        # generate eagle3 dataset using target model
        if args.is_vlm:
            image_grid_thw = (
                [thw.cuda().squeeze() for thw in data["image_grid_thw"]] if args.is_vlm else None
            )
            pixel_values = data["pixel_values"].cuda()
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
                is_vlm=args.is_vlm,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )
        tp_size = dist.get_world_size(get_tp_group()) if dist.is_initialized() else 1
        tp_rank = dist.get_rank(get_tp_group()) if dist.is_initialized() else 0
        def _shard(x):
            return x.chunk(tp_size, dim=0)[tp_rank] if tp_size > 1 else x
        input_ids = _shard(eagle3_data.input_ids)
        attention_mask = _shard(eagle3_data.attention_mask)
        loss_mask = _shard(eagle3_data.loss_mask)
        target = _shard(eagle3_data.target)
        hidden_states = _shard(eagle3_data.hidden_states)
    else:
        # offline: precomputed hidden states
        attention_mask = data["attention_mask"].cuda()
        hidden_states = data["hidden_state"].cuda()
        input_ids, target, loss_mask = target_model.preprocess(
            data["input_ids"], data["target"], data["loss_mask"]
        )
        input_ids = input_ids.cuda()
        target = target_model(target.cuda())
        loss_mask = loss_mask.cuda()

    plosses, _, acces = eagle3_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        target=target,
        hidden_states=hidden_states,
        position_ids=(data["position_ids"].cuda() if "position_ids" in data else None),
        image_grid_thw=image_grid_thw,
        is_vlm=args.is_vlm,
    )
    return plosses, acces


def main():
    parser, args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    init_distributed(
        timeout=args.dist_timeout, tp_size=args.tp_size, sp_ring_size=1, sp_ulysses_size=1
    )

    print_with_rank("Initialized distributed environment")
    is_online = args.eval_data_path is not None and args.eval_hidden_states_path is None

    # Build draft model
    draft_model_config, draft_model = build_draft_model(args)

    # Build target model
    target_model, processor = build_target_model(args, draft_model_config, is_online)

    # Build eval dataloader and vocab mapping
    eval_dataloader, vocab_mapping_path, _ = build_eval_dataloader(
        args, draft_model_config, processor
    )

    # Load vocab mapping
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # Build eagle3 model
    if args.is_vlm and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl" and args.tp_size == 1:
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        if is_online:
            eagle3_model = OnlineEagle3Model(
                target_model=target_model,
                draft_model=draft_model,
                length=args.ttt_length,
                attention_backend=args.attention_backend,
            )
        else:
            eagle3_model = OnlineEagle3Model(
                draft_model=draft_model,
                length=args.ttt_length,
                attention_backend=args.attention_backend,
            )

    # Evaluation loop
    eval_acces = [[] for _ in range(eagle3_model.length)]
    eval_plosses = [[] for _ in range(eagle3_model.length)]

    draft_model.eval()
    for data in tqdm(eval_dataloader, desc="Evaluating EAGLE3"):
        with torch.no_grad():
            plosses, acces = run_forward(args, eagle3_model, data, target_model, is_online)
            for i in range(len(acces)):
                eval_acces[i].append(acces[i])
            for i in range(len(plosses)):
                eval_plosses[i].append(plosses[i])

    # Aggregate per minibatch
    eval_acces = [torch.stack(acc).mean() if len(acc) > 0 else torch.tensor(0.0) for acc in eval_acces]
    eval_plosses = [torch.stack(pl).mean() if len(pl) > 0 else torch.tensor(0.0) for pl in eval_plosses]

    # Optional distributed average across ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        eval_acces_t = torch.stack(eval_acces).cuda()
        eval_plosses_t = torch.stack(eval_plosses).cuda()
        dist.all_reduce(eval_acces_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(eval_plosses_t, op=dist.ReduceOp.AVG)
        eval_acces = eval_acces_t.cpu().tolist()
        eval_plosses = eval_plosses_t.cpu().tolist()
    else:
        eval_acces = [x.item() for x in eval_acces]
        eval_plosses = [x.item() for x in eval_plosses]

    # Print results
    for i, (acc_i, ploss_i) in enumerate(zip(eval_acces, eval_plosses)):
        print_on_rank0(f"Eval position {i}: Acc={acc_i:.4f}, pLoss={ploss_i:.4f}")

    # Save results
    if dist.get_rank() == 0:
        result = {
            "ttt_length": int(eagle3_model.length),
            "eval_acc": {f"acc_{i}": float(v) for i, v in enumerate(eval_acces)},
            "eval_ploss": {f"ploss_{i}": float(v) for i, v in enumerate(eval_plosses)},
        }
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("Saved eval_results.json")

    destroy_distributed()


if __name__ == "__main__":
    main()