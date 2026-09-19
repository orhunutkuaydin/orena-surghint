"""Prepare and verify the SEGMENT model weights."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
METADATA = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
)


def references():
    return {
        path: digest
        for line in Path(__file__).with_name("checksums.txt").read_text().splitlines()
        if line and not line.startswith("#")
        for digest, path in [line.split(maxsplit=1)]
    }


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if file_digest(path) != expected:
        raise ValueError(f"Unexpected checkpoint or metadata: {path}")


def tensor_inventory(directory):
    """Index tensor bytes without loading the model into memory."""
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    mapping = (
        json.loads(index_path.read_text())["weight_map"]
        if index_path.is_file()
        else None
    )
    names = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
    inventory = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError("Unsafe model shard path")
        path = directory / name
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size < 100 << 20:
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(header_size))
        ranges = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in inventory or (mapping and mapping.get(key) != name):
                raise ValueError(f"Duplicate or incorrectly indexed tensor: {key}")
            start, end = value["data_offsets"]
            if start < 0 or end < start:
                raise ValueError("Invalid tensor offsets")
            inventory[key] = (
                path,
                8 + header_size + start,
                end - start,
                {"dtype": value["dtype"], "shape": value["shape"]},
            )
            ranges.append((start, end))
        offset = 0
        for start, end in sorted(ranges):
            if start != offset:
                raise ValueError("Noncontiguous or overlapping tensor data")
            offset = end
        if path.stat().st_size != 8 + header_size + offset:
            raise ValueError(f"Truncated or unexpected model data: {path}")
    if not inventory or (mapping and set(inventory) != set(mapping)):
        raise ValueError("Incomplete model index")
    return inventory


def model_digest(directory):
    """Hash names, shapes, dtypes and exact tensor bytes, independent of sharding."""
    digest = hashlib.sha256()
    for name, (path, start, remaining, metadata) in sorted(
        tensor_inventory(directory).items()
    ):
        digest.update(
            json.dumps([name, metadata], sort_keys=True, separators=(",", ":")).encode()
        )
        with path.open("rb") as stream:
            stream.seek(start)
            while remaining:
                block = stream.read(min(remaining, 8 << 20))
                if not block:
                    raise ValueError(f"Truncated tensor: {name}")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def verify_runtime(weights):
    expected = references()
    verify_file(
        weights / "custom/rfdetr-large.pth", expected["custom/rfdetr-large.pth"]
    )
    runtime = weights / "runtime/qwen"
    for name in METADATA:
        verify_file(runtime / name, expected[f"custom/qwen-lora/{name}"])
    if model_digest(runtime) != expected["runtime/qwen/tensor-content"]:
        raise ValueError("Merged Qwen tensor content differs from the release model")
    return runtime


def prepare(weights):
    """Reuse verified runtime weights or merge the pinned base and adapter once."""
    expected = references()
    verify_file(
        weights / "custom/rfdetr-large.pth", expected["custom/rfdetr-large.pth"]
    )
    runtime = weights / "runtime/qwen"
    if runtime.exists():
        return verify_runtime(weights)
    adapter = weights / "custom/qwen-lora"
    for name in ("adapter_config.json", "adapter_model.safetensors", *METADATA):
        verify_file(adapter / name, expected[f"custom/qwen-lora/{name}"])
    from huggingface_hub import snapshot_download

    base = weights / "base/qwen3.6-27b"
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=base,
        allow_patterns=[
            "*.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "generation_config.json",
        ],
    )
    for relative, digest in expected.items():
        if relative.startswith("base/"):
            verify_file(weights / relative, digest)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    temporary = runtime.with_name(f".qwen-{os.getpid()}")
    temporary.mkdir(parents=True)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        print(
            "Merging the Qwen adapter on CPU; this requires substantial RAM and disk space.",
            flush=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            base,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cpu",
        )
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise RuntimeError("Adapter remained unmerged")
        model.config.use_cache = True
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB")
        del model
        gc.collect()
        for name in METADATA:
            shutil.copyfile(adapter / name, temporary / name)
        if model_digest(temporary) != expected["runtime/qwen/tensor-content"]:
            raise ValueError(
                "Merged Qwen tensor content differs from the release model"
            )
        os.replace(temporary, runtime)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        torch.set_num_threads(threads)
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()
    (prepare if args.command == "prepare" else verify_runtime)(
        args.weights_dir.resolve()
    )
    print("SEGMENT weights verified.")


if __name__ == "__main__":
    main()
