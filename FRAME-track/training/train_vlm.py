#!/usr/bin/env python3
"""Train the Qwen3.5-9B vision-scope LoRA."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftConfig, get_peft_model
from PIL import Image, ImageFilter
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments

from surghint.inference import IMAGE_SIZE, SYSTEM_PROMPT

EXPECTED_TRAIN = 13_748
EXPECTED_TEST = 6_252
EXPECTED_QUESTIONS = 20_000
EXPECTED_FRAMES = 15_212
EXPECTED_HINT_CONTENT_SHA256 = (
    "51bc17b684fb919ced0207f71cbc55e2c6867f386023c4ad7f225c78fa2c02ee"
)


def hint_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("frame_id").to_dict("records"):
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
        )
    return digest.hexdigest()


def load_rows(root: Path, hints_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    train = pd.read_parquet(root / "metadata/train.parquet")
    test = pd.read_parquet(root / "metadata/test.parquet")
    if (len(train), len(test)) != (EXPECTED_TRAIN, EXPECTED_TEST):
        raise ValueError(f"question counts are {len(train)}/{len(test)}, expected 13748/6252")
    frame = pd.concat([train, test], ignore_index=True)
    if len(frame) != EXPECTED_QUESTIONS or frame["instance_id"].astype(str).nunique() != len(frame):
        raise ValueError("metadata must contain 20,000 unique questions")
    if frame["frame_id"].astype(str).nunique() != EXPECTED_FRAMES:
        raise ValueError("metadata must contain 15,212 unique frames")
    if set(train["frame_id"].astype(str)) & set(test["frame_id"].astype(str)):
        raise ValueError("train/test frame overlap")
    hint_frame = pd.read_parquet(hints_path)
    if hint_digest(hint_frame) != EXPECTED_HINT_CONTENT_SHA256:
        raise ValueError("hint content does not match the expected training artifact")
    if "hint" not in hint_frame or hint_frame["frame_id"].astype(str).duplicated().any():
        raise ValueError("hints must have unique frame_id and prerendered hint columns")
    hints = {str(row.frame_id): str(row.hint) for row in hint_frame.itertuples()}
    if set(hints) != set(frame["frame_id"].astype(str)):
        raise ValueError("hints do not cover exactly the 15,212 training frames")
    frame = frame.sort_values(["dataset", "split", "video", "id"], kind="stable")
    rows = []
    for row in frame.itertuples(index=False):
        image = root / str(row.image_path)
        if not image.is_file():
            raise FileNotFoundError(image)
        alternates = []
        for column in ("image_path_alt_1", "image_path_alt_1", "image_path_alt_3"):
            value = getattr(row, column, None)
            if isinstance(value, str) and value and (root / value).is_file():
                alternates.append(str(root / value))
        rows.append(
            {
                "image_path": str(image),
                "alt_paths": alternates,
                "frame_id": str(row.frame_id),
                "question": str(row.question),
                "answer": str(row.answer),
            }
        )
    return rows, hints


def appearance(image: Image.Image) -> Image.Image:
    if random.random() < 0.20:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 3.5)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if random.random() < 0.10:
        variance = random.uniform(0.0, 0.05)
        noise = np.random.default_rng(random.getrandbits(32)).normal(
            0.0, np.sqrt(variance) * max(float(array.std()), 1e-4), array.shape
        )
        array += noise.astype(np.float32)
    if random.random() < 0.15:
        array *= random.uniform(0.4, 1.25)
    if random.random() < 0.15:
        factor, mean = random.uniform(0.75, 1.25), array.mean()
        array = np.clip((array - mean) * factor + mean, array.min(), array.max())
    for probability, invert in ((0.10, True), (0.30, False)):
        if random.random() < probability:
            work = -array if invert else array
            mean, std = work.mean(), work.std()
            low, span = work.min(), max(float(work.max() - work.min()), 1e-7)
            work = np.power((work - low) / span, random.uniform(0.7, 1.5)) * span + low
            work = (work - work.mean()) / max(float(work.std()), 1e-7) * std + mean
            array = -work if invert else work
    return Image.fromarray((np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8))


class Collator:
    def __init__(self, processor: Any, hints: dict[str, str]) -> None:
        self.processor = processor
        self.hints = hints
        processor.tokenizer.padding_side = "right"

    def prompt(self, row: dict[str, Any]) -> tuple[str, Image.Image]:
        path = row["image_path"]
        if row["alt_paths"] and random.random() >= 0.5:
            path = random.choice(row["alt_paths"])
        with Image.open(path) as handle:
            image = appearance(handle.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC))
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"{self.hints[row['frame_id']]}\n{row['question']}",
                    },
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        if text.endswith("<think>\n"):
            text += "\n</think>\n\n"
        return text, image

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts, images = zip(*(self.prompt(row) for row in rows), strict=True)
        full = [
            prompt + row["answer"] + "<|im_end|>\n"
            for prompt, row in zip(prompts, rows, strict=True)
        ]
        batch = self.processor(text=full, images=list(images), padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index, (prompt, image) in enumerate(zip(prompts, images, strict=True)):
            length = self.processor(text=[prompt], images=[image], return_tensors="pt")[
                "input_ids"
            ].shape[1]
            labels[index, :length] = -100
        batch["labels"] = labels
        return batch


def lora_targets(model: torch.nn.Module) -> list[str]:
    linears = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    vision = [name for name in linears if "visual" in name.lower() or "vision" in name.lower()]
    language = [name for name in linears if name not in set(vision)]
    attention = (".self_attn.", ".linear_attn.", ".attn.", ".attention.")
    targets = [
        name for name in language if any(marker in name for marker in attention) and name != "lm_head"
    ]
    targets += [
        name
        for name in language
        if name.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj")
        and ".experts." not in name
    ]
    targets += vision
    return list(dict.fromkeys(targets))


class StopAt(TrainerCallback):
    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


class Losses(TrainerCallback):
    def __init__(self) -> None:
        self.values: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.values.append(float(logs["loss"]))


class AnswerOnlyTrainer(Trainer):
    """Compute the identical masked LM loss without materializing prompt logits."""

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        positions = torch.nonzero((labels[:, 1:] != -100).any(dim=0)).flatten()
        if not len(positions):
            raise RuntimeError("batch has no answer tokens")
        outputs = model(**inputs, logits_to_keep=positions)
        targets = labels[:, positions + 1].to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--hints", type=Path)
    parser.add_argument("--model", type=Path, default=Path("weights/base/qwen3.5-9b"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    root = args.data.resolve()
    hints_path = (args.hints or root / "metadata/hints.parquet").resolve()
    rows, hints = load_rows(root, hints_path)
    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLM
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoVLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets(model),
        ),
    )
    model.enable_input_require_grads()
    smoke = args.smoke_steps > 0
    if smoke:
        rows = rows[: max(args.smoke_steps, 4)]
        batch_size, grad_accum, stop = 1, 1, args.smoke_steps
        initial = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    else:
        batch_size, grad_accum, stop = args.batch_size, args.grad_accum, 1750
        initial = None
    losses = Losses()
    warmup = (
        {"warmup_ratio": 0.03}
        if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
        else {"warmup_steps": 0.03}
    )
    trainer = AnswerOnlyTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=3.0,
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            lr_scheduler_type="cosine",
            **warmup,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0 if smoke else args.num_workers,
            logging_steps=1 if smoke else 10,
            save_strategy="no" if smoke else "steps",
            save_steps=625,
            save_total_limit=8,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
            seed=0,
        ),
        train_dataset=rows,
        data_collator=Collator(processor, hints),
        callbacks=[StopAt(stop), losses],
    )
    result = trainer.train()
    adapter = args.output / "adapter"
    processor.tokenizer.padding_side = "left"
    model.save_pretrained(adapter)
    processor.save_pretrained(adapter)
    if trainer.state.global_step != stop:
        raise RuntimeError(f"stopped at step {trainer.state.global_step}, expected {stop}")
    if not losses.values or not all(np.isfinite(losses.values)):
        raise RuntimeError(f"non-finite or absent logged losses: {losses.values}")
    if smoke:
        changed = sum(
            not torch.equal(initial[name], parameter.detach().cpu())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if changed < 2:
            raise RuntimeError(f"only {changed} trainable tensors changed")
        PeftConfig.from_pretrained(adapter)
        report = {
            "status": "PASS",
            "scope": "training-path health check only",
            "global_step": trainer.state.global_step,
            "finite_losses": losses.values,
            "changed_trainable_tensors": changed,
            "adapter": str(adapter),
            "train_loss": float(result.training_loss),
        }
        (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
