# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HALO Stage-1 style hidden-state alignment training entry."""

# ruff: noqa: E402

import json
import os
import time
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List


os.environ["MODELING_BACKEND"] = "hf"

import torch
import torch.distributed as dist
from tqdm import trange
from transformers import AutoConfig

import hf_models  # noqa: F401
from in_place_ttt.ttt_aux.training import build_ttt_optimizer_param_groups
from tasks import train_torch as base
from tasks.eval_control import should_run_eval
from tasks.halo_hidden_alignment import (
    HiddenAlignmentOrchestrator,
    configure_hidden_alignment_trainable_params,
    resolve_hidden_align_layers,
)


logger = base.logger


def _build_transform(args, tokenizer):
    if args.data.data_type == "plaintext":
        return partial(
            base.process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    if args.data.data_type == "conversation":
        chat_template = base.build_chat_template(args.data.chat_template, tokenizer)
        return partial(
            base.process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    if args.data.data_type == "pretokenized":
        if base.process_pretokenized_example is None:
            raise NotImplementedError("Installed veomni package does not provide process_pretokenized_example.")
        return partial(
            base.process_pretokenized_example,
            input_ids_key=args.data.text_keys,
        )
    raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _pop_hidden_align_config(args) -> tuple[Dict[str, Any], Dict[str, Any]]:
    foundation = dict(args.model.foundation or {})
    align = {
        "teacher_path": (
            foundation.pop("hidden_align_teacher_path", None)
            or foundation.pop("distill_teacher_path", None)
            or args.model.model_path
        ),
        "loss_fn": str(foundation.pop("hidden_align_loss_fn", "mse")),
        "layers": foundation.pop("hidden_align_layers", None),
        "skip_ttt_layers": _as_bool(foundation.pop("hidden_align_skip_ttt_layers", False)),
        "train_scope": str(foundation.pop("hidden_align_train_scope", "layers")),
    }
    return foundation, align


def _teacher_foundation_config(teacher_path: str) -> Dict[str, Any]:
    teacher_config = AutoConfig.from_pretrained(teacher_path)
    num_layers = int(getattr(teacher_config, "num_hidden_layers"))
    return {
        "ttt_mode": False,
        "ttt_compress_window": 0,
        "ttt_aux_loss_weight": 0.0,
        "use_sliding_window": False,
        "sliding_window": None,
        "layer_types": ["full_attention"] * num_layers,
        "full_attention_layers": [],
    }


def _count_align_tokens(micro_batch: Dict[str, Any]) -> torch.Tensor:
    attention_mask = micro_batch.get("attention_mask")
    if attention_mask is not None:
        return torch.sum(attention_mask != 0)
    return torch.tensor(micro_batch["input_ids"].numel(), device=micro_batch["input_ids"].device)


def main():
    foundation_override = base._pop_dict_cli_arg("--model.foundation")

    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and base.is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    logger.info(f"Process_group timeout: {nccl_timeout}")
    dist.init_process_group(backend=base.get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = base.parse_args(base.Arguments)
    if foundation_override is not None:
        if args.model.foundation is None:
            args.model.foundation = {}
        args.model.foundation.update(foundation_override)

    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    base.get_torch_device().set_device(f"{base.get_device_type()}:{args.train.local_rank}")
    base.helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        base.helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        base.save_args(args, args.train.output_dir)

    student_foundation, align_cfg = _pop_hidden_align_config(args)

    Checkpointer = base.build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

    base.init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = base.build_tokenizer(args.model.tokenizer_path)
    transform = _build_transform(args, tokenizer)
    train_dataset = base._build_dataset_for_args(args, transform)
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    train_steps = base._compute_train_steps_compat(args, dataset_length)
    train_dataloader = base._build_dataloader_compat(args, train_dataset, train_steps)

    eval_dataloader = None
    eval_enable_multisource = False
    if args.data.eval_path and args.train.eval_steps > 0 and args.train.eval_batches > 0:
        eval_args = base._args_for_eval(args)
        eval_enable_multisource = eval_args.data.enable_multisource
        eval_dataset = base._build_dataset_for_args(eval_args, transform)
        eval_dataloader = base._build_dataloader_compat(eval_args, eval_dataset, args.train.eval_batches)

    logger.info_rank0("Prepare student model")
    student_model = base.build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        config_kwargs=student_foundation,
    )
    model_config = student_model.config
    align_layers = resolve_hidden_align_layers(
        model_config,
        align_cfg["layers"],
        skip_ttt_layers=align_cfg["skip_ttt_layers"],
    )
    trainable_params = configure_hidden_alignment_trainable_params(
        student_model,
        align_layers,
        train_scope=align_cfg["train_scope"],
    )
    logger.info_rank0(
        "HALO hidden alignment: "
        f"teacher={align_cfg['teacher_path']}, "
        f"loss_fn={align_cfg['loss_fn']}, "
        f"layers={align_layers}, "
        f"skip_ttt_layers={align_cfg['skip_ttt_layers']}, "
        f"train_scope={align_cfg['train_scope']}, "
        f"trainable_params={trainable_params:,}"
    )
    base.helper.print_device_mem_info("VRAM usage after building student")

    get_optimizer_pre_hook = getattr(student_model, "get_optimizer_pre_hook", None)
    student_model = base.build_parallelize_model(
        student_model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=student_model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    base.helper.print_device_mem_info("VRAM usage after FSDP-wrapping student")

    logger.info_rank0("Prepare frozen full-attention teacher model")
    teacher_model = base.build_foundation_model(
        config_path=args.model.config_path,
        weights_path=align_cfg["teacher_path"],
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        config_kwargs=_teacher_foundation_config(align_cfg["teacher_path"]),
    )
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    teacher_model.eval()
    teacher_model = base.build_parallelize_model(
        teacher_model,
        init_device=args.train.init_device,
        weights_path=align_cfg["teacher_path"],
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=teacher_model._no_split_modules + args.model.basic_modules,
        enable_reentrant=False,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    base.helper.print_device_mem_info("VRAM usage after FSDP-wrapping teacher")

    model = HiddenAlignmentOrchestrator(
        student=student_model,
        teacher=teacher_model,
        layer_idxs=align_layers,
        loss_fn=align_cfg["loss_fn"],
    )

    optimizer_param_groups = build_ttt_optimizer_param_groups(
        student_model,
        base_lr=args.train.lr,
        base_weight_decay=args.train.weight_decay,
        lr_multiplier=float(getattr(model_config, "ttt_param_lr_multiplier", 1.0)),
        weight_decay=getattr(model_config, "ttt_param_weight_decay", None),
    )
    if optimizer_param_groups is not None:
        ttt_group = optimizer_param_groups[-1]
        logger.info_rank0(
            "Using separate TTT optimizer group: "
            f"params={sum(param.numel() for param in ttt_group['params'])}, "
            f"lr={ttt_group['lr']:.2e}, weight_decay={ttt_group['weight_decay']}"
        )

    optimizer = base.build_optimizer(
        student_model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        param_groups=optimizer_param_groups,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(student_model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = base.build_lr_scheduler(
        optimizer,
        train_steps=train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            base.wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=base.wandb.Settings(console="off"),
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )

        if args.data.data_type in ["plaintext", "pretokenized"]:
            model_assets = [model_config, tokenizer]
        else:
            chat_template = base.build_chat_template(args.data.chat_template, tokenizer)
            model_assets = [model_config, chat_template]
        base.save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = base.helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter_kwargs = dict(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=getattr(args.train, "rmpad", False),
        rmpad_with_pos_ids=getattr(args.train, "rmpad_with_pos_ids", False),
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
        gc_steps=getattr(args.train, "gc_steps", 0),
    )
    environ_meter = base.helper.EnvironMeter(**base._filter_kwargs_for_callable(base.helper.EnvironMeter, environ_meter_kwargs))

    def save_training_checkpoint() -> None:
        nonlocal save_checkpoint_path
        base.helper.empty_cache()
        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
        state = {
            "model": student_model,
            "optimizer": optimizer,
            "extra_state": {
                "global_step": global_step,
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_dataloader": train_dataloader.state_dict(),
                "environ_meter": environ_meter.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
        dist.barrier()
        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    if args.train.load_checkpoint_path:
        state = {"model": student_model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // train_steps
        start_step = global_step % train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    base.helper.empty_cache()
    model_fwd_context, model_bwd_context = base.build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start HALO hidden alignment, train_steps: {train_steps}, "
        f"epochs: {args.train.num_train_epochs}"
    )

    stop_training = False
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                base.helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0.0
            base.synchronize()
            start_time = time.time()

            length_in_batch = torch.tensor(0, dtype=torch.int32, device=base.get_device_type())
            for micro_batch in micro_batches:
                length_in_batch += _count_align_tokens(micro_batch).to(length_in_batch.device)
            length_in_batch = base.all_reduce(length_in_batch, op="sum", group=base.get_parallel_state().fsdp_group)

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    base._strip_multisource_fields(micro_batch)

                micro_batch = base._move_micro_batch_to_device(micro_batch)
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)

                length_in_micro_batch = _count_align_tokens(micro_batch)
                loss_scale = length_in_micro_batch / length_in_batch * base.get_parallel_state().dp_size
                loss = model_outputs.loss * loss_scale

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = base.veomni_clip_grad_norm(student_model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = base.all_reduce((total_loss, grad_norm), group=base.get_parallel_state().fsdp_group)
            base.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"hidden_loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}",
                refresh=False,
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                train_metrics.update(
                    {
                        "training/loss": total_loss,
                        "training/hidden_align_loss": total_loss,
                        "training/hidden_align_layers": len(align_layers),
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                    }
                )
                logger.info_rank0(
                    f"[Step {global_step}] hidden_loss={total_loss:.4f}, "
                    f"grad_norm={grad_norm:.4f}, lr={lr:.2e}, "
                    f"tokens/s={train_metrics.get('tokens_per_second(M)', 0):.2f}M, "
                    f"mem={train_metrics.get('max_memory_allocated(GB)', 0):.1f}GB"
                )
                if args.train.use_wandb:
                    base.wandb.log(train_metrics, step=global_step)

            if eval_dataloader is not None and should_run_eval(
                args.data.eval_path,
                args.train.eval_steps,
                args.train.eval_batches,
                global_step,
            ):
                eval_loss = base._run_eval_loss(
                    student_model,
                    eval_dataloader,
                    args.train.eval_batches,
                    eval_enable_multisource,
                    model_fwd_context,
                )
                if eval_loss is not None and args.train.global_rank == 0:
                    logger.info_rank0(f"Eval loss at global_step {global_step}: {eval_loss:.4f}")
                    if args.train.use_wandb:
                        base.wandb.log({"eval/loss": eval_loss}, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                save_training_checkpoint()

            if args.train.stage_stop_steps and global_step >= args.train.stage_stop_steps:
                if save_checkpoint_path != os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}"):
                    save_training_checkpoint()
                logger.info_rank0(f"Reached stage_stop_steps={args.train.stage_stop_steps}; stop this stage.")
                stop_training = True
                break

        data_loader_tqdm.close()
        start_step = 0
        base.helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            save_training_checkpoint()
        if stop_training:
            break

    base.synchronize()
    base.helper.empty_cache()

    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = base.ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            ckpt_manager=args.train.ckpt_manager,
        )
        base.save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
