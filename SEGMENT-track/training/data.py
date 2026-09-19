"""Validated training records and answer-only supervision."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from . import config


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def local_file(root: Path, name: str) -> Path:
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path must remain inside the dataset directory: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def window(row: dict) -> list:
    request = row["request"]
    start, end = float(request["start_time"]), float(request["end_time"])
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError(f"Invalid clip times for {row['id']}")
    return [row["dataset"], request["videoID"], start, end]


def pipeline_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    names = ("detector.py", "hints.py", "tracking.py", "rendering.py", "video.py")
    return fingerprint({name: file_digest(root / "surghint" / name) for name in names})


def schedule(question_count: int) -> dict[str, int]:
    if question_count < 1 or question_count % config.WORLD_SIZE:
        raise ValueError("The question count must be positive and divisible by the number of GPUs")
    global_batch = config.WORLD_SIZE * config.BATCH_SIZE * config.GRAD_ACCUMULATION
    per_epoch = math.ceil(question_count / global_batch)
    return {
        "schedule_steps": 2 * per_epoch,
        "stop_after_steps": per_epoch,
        "warmup_steps": max(1, round(0.03 * 2 * per_epoch)),
    }


def load_manifest(root: Path) -> list[dict]:
    manifest = read_json(root / "manifest.json")
    if manifest["split"] != "train" or manifest["pipeline_sha256"] != pipeline_digest():
        raise ValueError("Training data use a different split or evidence pipeline; prepare the data again")
    rows = manifest["rows"]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Training records must have unique dataset-qualified IDs")
    if config.REQUIRE_FULL_TRAIN_SPLIT and len(rows) != config.TRAIN_QUESTIONS:
        raise ValueError(f"Expected {config.TRAIN_QUESTIONS} training questions, found {len(rows)}")
    checked: dict[Path, str] = {}
    for row in rows:
        if not isinstance(row["answer"], str) or not row["answer"].strip():
            raise ValueError(f"Missing answer: {row['id']}")
        request = row["request"]
        if not request["question"].strip() or not request["procedure_type"].strip():
            raise ValueError(f"Missing question or procedure: {row['id']}")
        if row["id"] != f"{row['dataset']}:train:{request['qID']}":
            raise ValueError(f"Inconsistent question ID: {row['id']}")
        for kind in ("clip", "tracking"):
            path = local_file(root, row[f"{kind}_path"])
            expected = row[f"{kind}_sha256"]
            if path not in checked:
                checked[path] = file_digest(path)
            if checked[path] != expected:
                raise ValueError(f"Prepared {kind} file changed: {path}")
        track = read_json(local_file(root, row["tracking_path"]))
        if track["window"] != window(row):
            raise ValueError(f"Tracking evidence belongs to a different clip: {row['id']}")
    return rows


def append_answer(batch: dict, tokenizer: Any, suffix: str) -> dict:
    """Append supervised tokens without retokenizing the visual prompt."""
    import torch

    ids = batch["input_ids"]
    tail = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(ids.device)
    if ids.shape[0] != 1 or tail.shape[0] != 1 or not tail.shape[1]:
        raise ValueError("Expected one nonempty answer")
    if ids.shape[1] + tail.shape[1] > config.MAX_TOKENS:
        raise ValueError("Training example exceeds the token budget")
    if "position_ids" in batch:
        raise ValueError("Explicit position IDs require a model-specific extension")
    result = batch.copy()
    result["input_ids"] = torch.cat((ids, tail), dim=1)
    result["attention_mask"] = torch.cat(
        (batch.get("attention_mask", torch.ones_like(ids)), torch.ones_like(tail)), dim=1
    )
    for name in ("token_type_ids", "mm_token_type_ids"):
        if name in result:
            result[name] = torch.cat((result[name], torch.zeros_like(tail, dtype=result[name].dtype)), dim=1)
    result["labels"] = torch.cat((torch.full_like(ids, -100), tail), dim=1)
    result["labels"][result["attention_mask"] == 0] = -100
    return result


class Collator:
    """Use the inference frame selection, rendering, and prompt for training."""

    def __init__(self, processor: Any, root: Path):
        self.processor = processor
        self.root = root
        self.readers: OrderedDict = OrderedDict()
        self.tracks: OrderedDict = OrderedDict()

    @staticmethod
    def cached(cache: OrderedDict, key: Path, loader: Any, limit: int):
        if key not in cache:
            if len(cache) >= limit:
                cache.popitem(last=False)
            cache[key] = loader(key)
        cache.move_to_end(key)
        return cache[key]

    def __call__(self, rows: list[dict]) -> dict:
        from types import SimpleNamespace

        from PIL import Image
        from surghint import prompts
        from surghint.video import open_video, visuals

        if len(rows) != 1:
            raise ValueError("SEGMENT training uses one example per device")
        row = rows[0]
        request = row["request"]
        reader = self.cached(self.readers, local_file(self.root, row["clip_path"]), open_video, 2)
        track = self.cached(self.tracks, local_file(self.root, row["tracking_path"]), read_json, 4)
        evidence, frames = visuals(reader, track, request)
        batch = prompts.encode(self.processor, request, evidence, frames)
        conversation = prompts.build_segment_conversation(
            [Image.fromarray(frame) for frame in frames], evidence["times"], SimpleNamespace(**request),
            prompts.SYSTEM_PROMPT_PREFIX + prompts.FO_DEFINITIONS + prompts.SYSTEM_PROMPT_SUFFIX + prompts.SYSTEM_NOTE,
            evidence["hint"],
        )
        prefix = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = self.processor.apply_chat_template(
            conversation + [{"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        if not full.startswith(prefix):
            raise ValueError("Training and inference chat templates disagree")
        return append_answer(batch, self.processor.tokenizer, full[len(prefix):])


if __name__ == "__main__":
    records = load_manifest(config.DATA_DIR)
    print(f"Validated {len(records)} training records")
