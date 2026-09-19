"""Merge a trained SEGMENT adapter into its pinned base model."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from . import config
from .data import atomic_json, file_digest
from surghint.weights import MODEL_ID, MODEL_REVISION


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = config.EXPORT_DIR.resolve()
    adapter = config.ADAPTER_DIR.resolve()
    if output.exists():
        raise FileExistsError(f"Export directory already exists: {output}")
    metadata = PeftConfig.from_pretrained(adapter)
    if metadata.r != config.LORA_RANK or metadata.lora_alpha != config.LORA_ALPHA:
        raise ValueError("Adapter rank or alpha differs from the training configuration")
    base = config.WEIGHTS_DIR / "base/qwen3.6-27b"
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=base, allow_patterns=[
        "*.json", "*.jinja", "*.txt", "*.model", "*.safetensors", "chat_templates/*",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".model-", dir=output.parent))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        model = AutoModelForImageTextToText.from_pretrained(
            base, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="cpu",
        )
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
        model.config.use_cache = True
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB")
        processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        processor.save_pretrained(temporary)
        atomic_json(temporary / "training_provenance.json", {
            "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
            "adapter_sha256": file_digest(adapter / "adapter_model.safetensors"),
            "precision": "bfloat16",
        })
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        torch.set_num_threads(threads)
    print(f"Merged model: {output}")


if __name__ == "__main__":
    main()
