#!/usr/bin/env python3
"""Download, validate, and build the exact runtime weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import detector as detector_loader
from .postprocessing import CLASSES

QWEN_MODEL = "Qwen/Qwen3.5-9B"
QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DINOV3_REPOSITORY = "https://github.com/facebookresearch/dinov3.git"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
PLAIN_DETR_REPOSITORY = "https://github.com/impiga/Plain-DETR.git"
PLAIN_DETR_COMMIT = "6ad930bb85f5d10417ebe979780132a9a466a8e0"

ADAPTER_SHA256 = "8603391bfe94a9a426ac3a56b92b1df3c1aa937d2059e68750923ca1f520c892"
ADAPTER_CONFIG_SHA256 = "0c3ce64c5286cc62a1b915128cf40856ef3518d5db5e66dde44c26e9c57f884c"
HEAD_SHA256 = "6982def5f80d23333ec7e1a1b376b31dcdf02c8784c65e14f4862f31eac7d45d"
DETECTOR_RUNTIME_SHA256 = "df29e34ecee1502fb108c29840cf370ddb4b45748552538058d7cba5ff758ae4"
BACKBONE_SHA256 = "a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7"
DETECTOR_BASE_SHA256 = "b0235ff7ea0a037b521f58eb7915d3a022c75319e93307dd48dfc5fd85c6f031"
QWEN_MERGE = "FP32 merge_and_unload, BF16 export"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path}: sha256 {actual}, expected {expected}")
    print(f"Verified {path}", flush=True)


def verify_qwen_runtime(output: Path) -> int:
    model = output / "model.safetensors"
    manifest_path = output / "MERGE_INFO.json"
    if not model.is_file():
        raise FileNotFoundError(model)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid Qwen merge manifest: {manifest_path}") from error
    expected = {
        "base_revision": QWEN_REVISION,
        "adapter_sha256": ADAPTER_SHA256,
        "merge": QWEN_MERGE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"{manifest_path}: {key} {manifest.get(key)!r}, expected {value!r}"
            )
    adapter_config = manifest.get("adapter_config_sha256")
    if adapter_config is not None and adapter_config != ADAPTER_CONFIG_SHA256:
        raise RuntimeError(
            f"{manifest_path}: adapter_config_sha256 {adapter_config!r}, "
            f"expected {ADAPTER_CONFIG_SHA256!r}"
        )
    actual_size = model.stat().st_size
    if actual_size <= 0:
        raise RuntimeError(f"empty Qwen runtime model: {model}")
    recorded_size = manifest.get("model_size_bytes")
    if recorded_size is not None and recorded_size != actual_size:
        raise RuntimeError(f"{model}: size {actual_size}, manifest records {recorded_size}")
    print(f"Verified {model}", flush=True)
    return actual_size


def clone_pinned(url: str, commit: str, output: Path) -> None:
    if not (output / ".git").is_dir():
        if output.exists():
            raise RuntimeError(f"refusing non-git source directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", url, str(output)], check=True)
    dirty = subprocess.run(
        ["git", "-C", str(output), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"tracked changes in {output}:\n{dirty}")
    subprocess.run(["git", "-C", str(output), "fetch", "--depth", "1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(output), "checkout", "--detach", commit], check=True)
    actual = subprocess.run(
        ["git", "-C", str(output), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"{output}: commit {actual}, expected {commit}")


def verify_source(source: Path, commit: str) -> None:
    if not (source / "hubconf.py").is_file():
        raise FileNotFoundError(f"missing DINOv3 hubconf.py: {source}")
    if (source / ".git").is_dir():
        actual = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError(f"tracked changes in pinned source: {source}")
    else:
        marker = source / "COMMIT"
        actual = marker.read_text().strip() if marker.is_file() else ""
    if actual != commit:
        raise RuntimeError(f"{source}: commit {actual or 'unknown'}, expected {commit}")


def verify_qwen_revision(base: Path) -> None:
    marker = base / ".revision"
    candidates = {base.resolve().name}
    if marker.is_file():
        candidates.add(marker.read_text().strip())
    if QWEN_REVISION not in candidates:
        raise RuntimeError(
            f"cannot verify Qwen revision {QWEN_REVISION} at {base}; "
            "use the download-qwen command or a snapshot directory named by the revision"
        )


def download_qwen(weights: Path) -> None:
    from huggingface_hub import snapshot_download

    output = weights / "base/qwen3.5-9b"
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(QWEN_MODEL, revision=QWEN_REVISION, local_dir=output)
    (output / ".revision").write_text(QWEN_REVISION + "\n")
    verify_qwen_revision(output)


def fix_dtype(value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if key == "dtype" and item == "float32":
            value[key] = "bfloat16"
        else:
            fix_dtype(item)


def build_qwen(weights: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLM

    base = weights / "base/qwen3.5-9b"
    adapter = weights / "custom/qwen-lora"
    output = weights / "runtime/qwen"
    verify_qwen_revision(base)
    require_hash(adapter / "adapter_model.safetensors", ADAPTER_SHA256)
    require_hash(adapter / "adapter_config.json", ADAPTER_CONFIG_SHA256)
    if output.exists():
        verify_qwen_runtime(output)
        return
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    previous_threads = torch.get_num_threads()
    try:
        # Preserve the CPU reduction order used for the byte-exact release merge.
        torch.set_num_threads(14)
        print("Merging Qwen and LoRA in FP32 on CPU; exporting BF16 weights.", flush=True)
        model = AutoVLM.from_pretrained(
            base,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = model.merge_and_unload().to(torch.bfloat16)
        if any("lora" in name.lower() for name, _ in model.named_modules()):
            raise RuntimeError("LoRA module remained after merge")
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="60GB")
        AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(temporary)
        config_path = temporary / "config.json"
        config = json.loads(config_path.read_text())
        fix_dtype(config)
        config["use_cache"] = True
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        runtime_size = (temporary / "model.safetensors").stat().st_size
        (temporary / "MERGE_INFO.json").write_text(
            json.dumps(
                {
                    "base_model": QWEN_MODEL,
                    "base_revision": QWEN_REVISION,
                    "adapter_sha256": ADAPTER_SHA256,
                    "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
                    "merge": QWEN_MERGE,
                    "model_file": "model.safetensors",
                    "model_size_bytes": runtime_size,
                },
                indent=2,
            )
            + "\n"
        )
        verify_qwen_runtime(temporary)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        torch.set_num_threads(previous_threads)


def load_head(path: Path) -> dict:
    import torch

    require_hash(path, HEAD_SHA256)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if int(state.get("epoch", -1)) != detector_loader.EXPECTED_HEAD_EPOCH:
        raise RuntimeError("detector head epoch is not 10")
    if list(state.get("classes", ())) != list(CLASSES):
        raise RuntimeError("detector head taxonomy differs")
    trainable = state.get("trainable")
    if not isinstance(trainable, dict) or not trainable:
        raise RuntimeError("detector head has no trainable state")
    if any(tensor.dtype != torch.float32 for tensor in trainable.values()):
        raise RuntimeError("detector head contains a non-FP32 tensor")
    return trainable


def detector_state(backbone: dict, head: dict) -> dict:
    """Combine the frozen backbone and complete trained head, including aliases."""
    import torch

    architecture = detector_loader.build_architecture()
    schema = architecture.state_dict()
    prefix = "detector.backbone.0._backbone.backbone."
    names = {
        id(parameter): name
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    }
    if set(head) != set(names.values()):
        raise RuntimeError("detector head does not exactly cover trainable parameters")
    if set(backbone) != {name.removeprefix(prefix) for name in schema if name.startswith(prefix)}:
        raise RuntimeError("backbone keys differ from the pinned DINOv3 architecture")
    state = {}
    for name, template in schema.items():
        if name.startswith(prefix):
            tensor = backbone[name.removeprefix(prefix)].to(dtype=torch.bfloat16, copy=True)
        else:
            canonical = names[id(architecture.get_parameter(name))]
            tensor = head[canonical].detach().clone()
        if tensor.shape != template.shape:
            raise RuntimeError(f"unexpected tensor shape: {name}")
        state[name] = tensor
    return state


def build_detector_from_sources(weights: Path) -> None:
    import torch

    source = weights / "base/dinov3/source"
    models = weights / "base/dinov3"
    verify_source(source, DINOV3_COMMIT)
    os.environ[detector_loader.DINOV3_SOURCE_ENV] = str(source)
    head_path = weights / "custom/plain-detr-head.pth"
    head = load_head(head_path)
    output_dir = weights / "runtime/detector"
    if output_dir.exists():
        require_hash(output_dir / detector_loader.SERVING_STATE_NAME, DETECTOR_RUNTIME_SHA256)
        return
    require_hash(models / detector_loader.BACKBONE_WEIGHTS, BACKBONE_SHA256)
    print("Building Plain-DETR with the frozen DINOv3 backbone on CPU.", flush=True)
    backbone = torch.load(
        models / detector_loader.BACKBONE_WEIGHTS,
        map_location="cpu", mmap=True, weights_only=True,
    )
    state = detector_state(backbone, head)
    del backbone
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    temporary_dir.mkdir(parents=True)
    try:
        temporary = temporary_dir / (detector_loader.SERVING_STATE_NAME + ".tmp")
        torch.save(state, temporary)
        require_hash(temporary, DETECTOR_RUNTIME_SHA256)
        os.replace(temporary, temporary_dir / detector_loader.SERVING_STATE_NAME)
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def prepare_for_inference(weights: Path) -> None:
    """Prepare missing runtime files once; never replace an existing model."""
    need_detector = not (weights / "runtime/detector/serving_state.pth").is_file()
    need_qwen = not (weights / "runtime/qwen/model.safetensors").is_file()
    if need_detector:
        require_hash(weights / "custom/plain-detr-head.pth", HEAD_SHA256)
        require_hash(weights / "base/dinov3" / detector_loader.BACKBONE_WEIGHTS, BACKBONE_SHA256)
    if need_qwen:
        require_hash(weights / "custom/qwen-lora/adapter_model.safetensors", ADAPTER_SHA256)
        require_hash(weights / "custom/qwen-lora/adapter_config.json", ADAPTER_CONFIG_SHA256)
    source = weights / "base/dinov3/source"
    if not source.exists():
        clone_pinned(DINOV3_REPOSITORY, DINOV3_COMMIT, source)
    verify_source(source, DINOV3_COMMIT)
    if need_detector:
        build_detector_from_sources(weights)
    if need_qwen:
        if not (weights / "base/qwen3.5-9b").exists():
            download_qwen(weights)
        build_qwen(weights)
    gc.collect()


def verify(weights: Path) -> None:
    verify_qwen_revision(weights / "base/qwen3.5-9b")
    verify_source(weights / "base/dinov3/source", DINOV3_COMMIT)
    require_hash(weights / "custom/qwen-lora/adapter_model.safetensors", ADAPTER_SHA256)
    require_hash(weights / "custom/qwen-lora/adapter_config.json", ADAPTER_CONFIG_SHA256)
    require_hash(weights / "custom/plain-detr-head.pth", HEAD_SHA256)
    verify_qwen_runtime(weights / "runtime/qwen")
    require_hash(weights / "runtime/detector/serving_state.pth", DETECTOR_RUNTIME_SHA256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-qwen")
    sources = commands.add_parser("sources", help="fetch pinned model source code")
    sources.add_argument("--training", action="store_true", help="also fetch detector training code")
    commands.add_parser("qwen")
    commands.add_parser("detector")
    commands.add_parser("verify")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights_dir.resolve()
    if args.command == "download-qwen":
        download_qwen(weights)
    elif args.command == "sources":
        clone_pinned(DINOV3_REPOSITORY, DINOV3_COMMIT, weights / "base/dinov3/source")
        if args.training:
            clone_pinned(
                PLAIN_DETR_REPOSITORY, PLAIN_DETR_COMMIT,
                weights / "base/dinov3/plain-detr-source",
            )
    elif args.command == "qwen":
        build_qwen(weights)
    elif args.command == "detector":
        build_detector_from_sources(weights)
    elif args.command == "verify":
        verify(weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
