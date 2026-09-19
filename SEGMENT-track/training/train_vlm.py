"""Train the SEGMENT language adapter with two data-parallel GPUs."""

from __future__ import annotations

import math
import os
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import config
from .data import Collator, atomic_json, file_digest, load_manifest, read_json, schedule
from surghint.weights import MODEL_ID, MODEL_REVISION

LORA_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)


def lora_targets(model) -> list[str]:
    import torch

    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "language_model" in name
        and name.rsplit(".", 1)[-1] in LORA_SUFFIXES
    ]
    found = {name.rsplit(".", 1)[-1] for name in targets}
    if set(LORA_SUFFIXES) - found or any("visual" in name or "vision" in name for name in targets):
        raise ValueError("The model does not expose the expected language-only LoRA targets")
    return targets


def check_kernels() -> None:
    from flash_attn import flash_attn_func
    from transformers.models.qwen3_5 import modeling_qwen3_5

    required = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
    if not callable(flash_attn_func) or not all(callable(getattr(modeling_qwen3_5, name, None)) for name in required):
        raise RuntimeError("Training requires FlashAttention 2, causal-conv1d, and Flash Linear Attention")
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError("The Qwen linear-attention CUDA kernels are unavailable")


def recipe(rows: list[dict]) -> dict:
    return {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "manifest_sha256": file_digest(config.DATA_DIR / "manifest.json"),
        "questions": len(rows), "frames": 24, "width": 768, "height": 448,
        "world_size": config.WORLD_SIZE, "per_device_batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION,
        "learning_rate": config.LEARNING_RATE, "lora_rank": config.LORA_RANK,
        "lora_alpha": config.LORA_ALPHA, "lora_dropout": config.LORA_DROPOUT,
        "lora_suffixes": list(LORA_SUFFIXES), "seed": config.SEED,
        "attention_backend": config.ATTENTION_BACKEND, "schedule_epochs": 2,
        "training_epochs": 1, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "optimizer": "adamw_torch_fused", "num_workers": config.NUM_WORKERS,
        "source_sha256": {
            name: file_digest(Path(__file__).with_name(name)) for name in ("data.py", "train_vlm.py")
        },
        "prompt_sha256": file_digest(config.ROOT / "surghint/prompts.py"),
        **schedule(len(rows)),
    }


def main() -> None:
    import fcntl

    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainerState, TrainingArguments, set_seed

    if config.WORLD_SIZE != 2 or config.BATCH_SIZE != 1 or config.GRAD_ACCUMULATION != 4:
        raise ValueError("This recipe uses two GPUs, batch size one, and accumulation four")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available() or world != config.WORLD_SIZE or torch.cuda.device_count() != config.WORLD_SIZE:
        raise RuntimeError("Launch with torchrun --standalone --nproc_per_node=2 -m training.train_vlm")
    if config.NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be nonnegative")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPUs are required")
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    rank = dist.get_rank()
    lock = None
    try:
        set_seed(config.SEED)
        check_kernels()
        rows = load_manifest(config.DATA_DIR)
        settings = recipe(rows)
        output = config.OUTPUT_DIR.resolve()
        resume = config.RESUME_CHECKPOINT.resolve() if config.RESUME_CHECKPOINT else None
        saved_state = None
        if resume:
            if not resume.is_relative_to(output / "checkpoints") or not resume.is_dir():
                raise ValueError("RESUME_CHECKPOINT must be a checkpoint inside this output directory")
            if read_json(output / "training_config.json") != settings:
                raise ValueError("Training data or settings changed since the checkpoint was saved")
            saved_state = TrainerState.load_from_json(str(resume / "trainer_state.json"))
            if saved_state.global_step > settings["stop_after_steps"]:
                raise ValueError("The checkpoint exceeds the one-epoch stopping point")
        elif rank == 0 and output.exists() and any(output.iterdir()):
            raise FileExistsError("OUTPUT_DIR is not empty; choose a new directory or set RESUME_CHECKPOINT")
        base = config.WEIGHTS_DIR / "base/qwen3.6-27b"
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            lock = (output / ".training.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(output / "training_config.json", settings)
            snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=base, allow_patterns=[
                "*.json", "*.jinja", "*.txt", "*.model", "*.safetensors", "chat_templates/*",
            ])
            (base / ".revision").write_text(MODEL_REVISION + "\n", encoding="utf-8")
            packages = ("torch", "torchvision", "transformers", "peft", "accelerate", "decord", "flash-attn", "causal-conv1d", "flash-linear-attention")
            versions = {}
            for name in (*packages, "fla-core"):
                try:
                    versions[name] = version(name)
                except PackageNotFoundError:
                    versions[name] = None
            atomic_json(output / "environment.json", versions)
        dist.barrier()
        processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        processor.tokenizer.padding_side = "right"
        model = AutoModelForImageTextToText.from_pretrained(
            base, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation=config.ATTENTION_BACKEND, device_map={"": local_rank},
        )
        for model_config in (model.config, model.config.text_config, model.config.vision_config):
            if model_config._attn_implementation != config.ATTENTION_BACKEND:
                raise RuntimeError("The model did not enable the requested attention backend")
        targets = lora_targets(model)
        model.config.use_cache = False
        finished = saved_state is not None and saved_state.global_step == settings["stop_after_steps"]
        if finished:
            model = PeftModel.from_pretrained(model, resume, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                target_modules=targets, bias="none", task_type="CAUSAL_LM",
            ))
        model.enable_input_require_grads()

        class StopAfterEpoch(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if state.max_steps != settings["schedule_steps"]:
                    raise ValueError("The optimizer schedule does not match the two-epoch horizon")

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= settings["stop_after_steps"]:
                    control.should_training_stop = True
                    control.should_save = True
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
                    raise RuntimeError("Training produced a non-finite loss")

        worker_options = {"dataloader_persistent_workers": True, "dataloader_prefetch_factor": 2} if config.NUM_WORKERS else {}
        trainer = Trainer(
            model=model, processing_class=processor, train_dataset=rows,
            data_collator=Collator(processor, config.DATA_DIR), callbacks=[StopAfterEpoch()],
            args=TrainingArguments(
                output_dir=str(output / "checkpoints"), num_train_epochs=2,
                learning_rate=config.LEARNING_RATE, per_device_train_batch_size=1,
                gradient_accumulation_steps=4, lr_scheduler_type="cosine",
                warmup_steps=settings["warmup_steps"], weight_decay=0.01, max_grad_norm=1.0,
                optim="adamw_torch_fused", bf16=True, tf32=True, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_num_workers=config.NUM_WORKERS, dataloader_pin_memory=True,
                dataloader_drop_last=False, logging_steps=10, logging_first_step=True,
                logging_nan_inf_filter=False, save_strategy="steps", save_steps=400, save_total_limit=3,
                eval_strategy="no", load_best_model_at_end=False, remove_unused_columns=False,
                label_names=["labels"], report_to="none", ddp_find_unused_parameters=False,
                ddp_timeout=2700, average_tokens_across_devices=True, seed=config.SEED,
                data_seed=config.SEED, **worker_options,
            ),
        )
        if finished:
            trainer.state = saved_state
        else:
            trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if trainer.state.global_step != settings["stop_after_steps"] or not math.isclose(trainer.state.epoch, 1.0, abs_tol=1e-6):
            raise RuntimeError("Training did not finish at the expected one-epoch checkpoint")
        trainer.accelerator.wait_for_everyone()
        if rank == 0:
            adapter = output / "adapter"
            trainer.accelerator.unwrap_model(trainer.model).save_pretrained(adapter, safe_serialization=True)
            processor.tokenizer.padding_side = "left"
            processor.save_pretrained(adapter)
            atomic_json(output / "lora_targets.json", targets)
            atomic_json(output / "completed.json", {"global_step": trainer.state.global_step, "epoch": trainer.state.epoch})
            print(f"Adapter saved to {adapter}", flush=True)
        dist.barrier()
    finally:
        if lock:
            lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
