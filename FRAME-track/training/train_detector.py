#!/usr/bin/env python3
"""Train an eight-class Plain-DETR head on public SurgHint HeiCo annotations."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from surghint.detr_utils import (
    enable_bf16_frozen_backbone,
    replace_class_heads,
)
from surghint.postprocessing import CLASSES

PLAIN_DETR_COMMIT = "6ad930bb85f5d10417ebe979780132a9a466a8e0"


def xyxy_to_cxcywh(
    boxes: Any,
    *,
    width: int,
    height: int,
    normalize: bool,
    min_size: float = 1.0,
):
    """Convert resized xyxy boxes to center/extent coordinates.

    Clamp extents to ``min_size`` before division and logarithms in the
    box-regression loss. Normalize to image dimensions when requested.
    """

    import torch

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    converted = torch.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            (boxes[:, 1] + boxes[:, 3]) / 2.0,
            (boxes[:, 2] - boxes[:, 0]).clamp(min=min_size),
            (boxes[:, 3] - boxes[:, 1]).clamp(min=min_size),
        ],
        dim=1,
    )
    if normalize:
        converted = converted / converted.new_tensor([width, height, width, height])
    return converted


def enable_hybrid_queries(detector: Any) -> int:
    """Restore one-to-many queries disabled by the inference hub loader.

    The query embedding table retains the full training capacity. Its size
    determines the total query and proposal counts for hybrid training.
    """

    one2one = int(detector.num_queries_one2one)
    query_embed = getattr(detector, "query_embed", None)
    if query_embed is None or not hasattr(query_embed, "num_embeddings"):
        raise RuntimeError(
            "detector has no query_embed table — cannot recover the one2many "
            "query count from an inference-truncated model"
        )
    total = int(query_embed.num_embeddings)
    if total <= one2one:
        raise RuntimeError(
            f"query_embed holds {total} embeddings and one2one claims "
            f"{one2one} — no one2many capacity to enable"
        )
    detector.num_queries = total
    detector.transformer.two_stage_num_proposals = total
    return total


def stabilize_bbox2delta(
    box_ops: Any,
    *,
    consumers: Sequence[Any] = (),
    ref_floor: float = 4.0,
    delta_clamp: float = 16.0,
) -> bool:
    """Bound detached box-regression targets for small objects."""

    import torch

    applied = not getattr(box_ops.bbox2delta, "_stabilized", False)
    if applied:
        original = box_ops.bbox2delta

        def stabilized(proposals, gt, *args, **kwargs):
            # Detached, out-of-place targets avoid modifying views saved by autograd.
            centers = proposals[..., :2].detach().float()
            extents = proposals[..., 2:].detach().float().clamp(min=ref_floor)
            safe_proposals = torch.cat((centers, extents), dim=-1)
            safe_gt = gt.detach().float()
            deltas = original(safe_proposals, safe_gt, *args, **kwargs)
            return deltas.clamp(min=-delta_clamp, max=delta_clamp)

        stabilized._stabilized = True
        box_ops.bbox2delta = stabilized

    # Update direct imports as well as the box_ops module attribute.
    for consumer in consumers:
        if hasattr(consumer, "bbox2delta"):
            consumer.bbox2delta = box_ops.bbox2delta
    return applied


def hybrid_loss(
    criterion: Any,
    outputs: dict[str, Any],
    targets: Sequence[dict[str, Any]],
    *,
    k_one2many: int = 6,
    lambda_one2many: float = 1.0,
):
    """Compute the combined one-to-one and one-to-many Plain-DETR loss."""

    loss_dict = criterion(outputs, targets)
    logits = outputs.get("pred_logits_one2many")
    query_count = int(logits.shape[1]) if logits is not None else 0
    if query_count > 0:
        one2many = {
            key[: -len("_one2many")]: value
            for key, value in outputs.items()
            if key.endswith("_one2many")
        }
        repeated = [
            {
                "boxes": target["boxes"].repeat(k_one2many, 1),
                "labels": target["labels"].repeat(k_one2many),
            }
            for target in targets
        ]
        for key, value in criterion(one2many, repeated).items():
            loss_dict[f"{key}_one2many"] = value * lambda_one2many

    weights = criterion.weight_dict
    weighted = [loss_dict[key] * weights[key] for key in loss_dict if key in weights]
    if not weighted:
        raise RuntimeError("criterion returned no losses present in its weight_dict")
    return sum(weighted), loss_dict


def read_coco(root: Path) -> dict:
    path = root / "_annotations.coco.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def audit(root: Path, check_images: bool = True) -> dict:
    payload = read_coco(root)
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    if set(categories.values()) != set(CLASSES):
        raise ValueError("COCO taxonomy differs from the eight release classes")
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not images or not annotations:
        raise ValueError("public HeiCo COCO data must contain images and boxes")
    image_ids = [int(item["id"]) for item in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate COCO image IDs")
    known = set(image_ids)
    if any(str(item.get("source_kind")) != "surghint_heico" for item in images):
        raise ValueError("this entry point accepts only SurgHint HeiCo data")
    for item in annotations:
        if int(item["image_id"]) not in known or int(item["category_id"]) not in categories:
            raise ValueError("invalid COCO annotation reference")
    if check_images:
        missing = [item["file_name"] for item in images if not (root / item["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} image(s), first: {missing[0]}")
    return {
        "status": "PASS",
        "scope": "public SurgHint HeiCo annotations only",
        "images": len(images),
        "annotations": len(annotations),
        "classes": list(CLASSES),
    }


def dataset(root: Path, short_side: int):
    import torch
    import torchvision.transforms.functional as tvf
    from PIL import Image
    from torch.utils.data import Dataset

    class CocoBoxes(Dataset):
        def __init__(self) -> None:
            payload = read_coco(root)
            ids = {str(item["name"]): int(item["id"]) for item in payload["categories"]}
            labels = {ids[name]: index for index, name in enumerate(CLASSES)}
            self.by_image: dict[int, list[tuple[dict, int]]] = {}
            for item in payload["annotations"]:
                label = labels.get(int(item["category_id"]))
                if label is not None:
                    self.by_image.setdefault(int(item["image_id"]), []).append((item, label))
            positives = set(self.by_image)
            self.images = [
                item
                for item in payload["images"]
                if int(item["id"]) in positives or bool(item.get("allow_empty"))
            ]

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int):
            entry = self.images[index]
            with Image.open(root / entry["file_name"]) as handle:
                image = handle.convert("RGB")
            width, height = image.size
            scale = short_side / min(width, height)
            new_width, new_height = round(width * scale), round(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            boxes, labels = [], []
            for item, label in self.by_image.get(int(entry["id"]), []):
                x, y, box_width, box_height = item["bbox"]
                boxes.append(
                    [x * scale, y * scale, (x + box_width) * scale, (y + box_height) * scale]
                )
                labels.append(label)
            boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor(labels, dtype=torch.int64)
            if torch.rand(()) < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if len(boxes):
                    left = new_width - boxes[:, 2].clone()
                    right = new_width - boxes[:, 0].clone()
                    boxes[:, 0], boxes[:, 2] = left, right
            tensor = tvf.normalize(
                tvf.to_tensor(image),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            return tensor, {
                "boxes": xyxy_to_cxcywh(
                    boxes, width=new_width, height=new_height, normalize=False
                ),
                "labels": labels,
            }

    return CocoBoxes()


def collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def verify_source(path: Path) -> None:
    import subprocess

    actual = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != PLAIN_DETR_COMMIT:
        raise RuntimeError(f"{path}: commit {actual}, expected {PLAIN_DETR_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"tracked changes in {path}")


def criterion(source: Path, device: str):
    verify_source(source)
    sys.path.insert(0, str(source))
    criterion_module = matcher_module = None
    for candidate in sorted((source / "models").glob("*.py")):
        text = candidate.read_text()
        if criterion_module is None and "class SetCriterion" in text:
            criterion_module = f"models.{candidate.stem}"
        if matcher_module is None and "def build_matcher" in text:
            matcher_module = f"models.{candidate.stem}"
    if not criterion_module or not matcher_module:
        raise RuntimeError("Plain-DETR criterion or matcher not found")
    crit_mod = importlib.import_module(criterion_module)
    match_mod = importlib.import_module(matcher_module)
    box_ops = importlib.import_module("util.box_ops")
    stabilize_bbox2delta(box_ops, consumers=(match_mod,))
    matcher = match_mod.build_matcher(
        argparse.Namespace(
            set_cost_class=2.0,
            set_cost_bbox=1.0,
            set_cost_giou=2.0,
            reparam=True,
            focal_alpha=0.25,
        )
    )
    base = {"loss_ce": 2.0, "loss_bbox": 1.0, "loss_giou": 2.0}
    weights = dict(base)
    for layer in range(5):
        weights.update({f"{key}_{layer}": value for key, value in base.items()})
    weights.update({f"{key}_enc": value for key, value in base.items()})
    weights.update({f"{key}_one2many": value for key, value in dict(weights).items()})
    candidates = [
        value
        for name, value in vars(crit_mod).items()
        if isinstance(value, type) and name.startswith("SetCriterion")
    ]
    cls = candidates[-1]
    known = {
        "num_classes": len(CLASSES),
        "matcher": matcher,
        "weight_dict": weights,
        "losses": ["labels", "boxes", "cardinality"],
        "focal_alpha": 0.25,
        "reparam": True,
    }
    signature = inspect.signature(cls.__init__)
    kwargs = {name: value for name, value in known.items() if name in signature.parameters}
    missing = [
        item.name
        for item in signature.parameters.values()
        if item.name != "self" and item.default is inspect.Parameter.empty and item.name not in kwargs
    ]
    if missing:
        raise RuntimeError(f"unsupported Plain-DETR criterion signature: {missing}")
    result = cls(**kwargs)
    result.weight_dict = weights
    return result.to(device)


def model(weights: Path, device: str):
    from surghint import detector as detector_loader
    import torch
    from surghint.weights import (
        BACKBONE_SHA256, DETECTOR_BASE_SHA256, DINOV3_COMMIT,
        require_hash, verify_source as verify_backbone_source,
    )

    source = weights / "base/dinov3/source"
    verify_backbone_source(source, DINOV3_COMMIT)
    require_hash(weights / "base/dinov3" / detector_loader.BACKBONE_WEIGHTS, BACKBONE_SHA256)
    require_hash(weights / "base/dinov3" / detector_loader.DETECTOR_WEIGHTS, DETECTOR_BASE_SHA256)
    os.environ[detector_loader.DINOV3_SOURCE_ENV] = str(source)
    os.environ.setdefault("TORCH_HOME", str(weights / "base/dinov3/torch_home"))
    detector_loader._seed_hub_checkpoint_cache(weights / "base/dinov3")
    with detector_loader._mmap_torch_load():
        wrapper = detector_loader._hub_load(
            weights=str(weights / "base/dinov3" / detector_loader.DETECTOR_WEIGHTS),
            backbone_weights=str(weights / "base/dinov3" / detector_loader.BACKBONE_WEIGHTS),
        )
    replace_class_heads(wrapper, num_classes=len(CLASSES))
    enable_bf16_frozen_backbone(wrapper.detector)
    wrapper = wrapper.to(device)
    wrong = [
        name
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if wrong:
        raise RuntimeError(f"non-FP32 trainable detector tensors: {wrong[:5]}")
    enable_hybrid_queries(wrapper.detector)
    return wrapper


def trainable_state(wrapper):
    wanted = {name for name, parameter in wrapper.named_parameters() if parameter.requires_grad}
    return {name: tensor for name, tensor in wrapper.state_dict().items() if name in wanted}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data.resolve()
    report = audit(root, not args.skip_image_check)
    if args.dry_run:
        report.update(
            {
                "short_side": 1024,
                "backbone": "frozen BF16 DINOv3 ViT-7B/16",
                "head": "FP32 eight-class Plain-DETR",
                "optimizer": "AdamW",
                "learning_rate": 2e-5,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.999],
                "epochs": args.epochs,
                "warmup_steps": 1000,
            }
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a BF16-capable CUDA GPU is required")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    weights = args.weights_dir.resolve()
    wrapper = model(weights, "cuda")
    loss_fn = criterion(weights / "base/dinov3/plain-detr-source", "cuda")
    loader = DataLoader(
        dataset(root, 1024),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=True,
    )
    if not len(loader):
        raise ValueError("dataset is smaller than one batch; reduce --batch-size")
    parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=1e-4, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(1000, max(1, total_steps - 1))

    def schedule(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 0.01)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        wrapper.train()
        loss_fn.train()
        optimizer.zero_grad()
        started = time.monotonic()
        for step, (images, targets) in enumerate(loader):
            outputs = wrapper.detector([image.to("cuda") for image in images])
            required = ("pred_logits", "pred_boxes", "pred_boxes_old", "pred_deltas")
            wrong = [key for key in required if key in outputs and outputs[key].dtype != torch.float32]
            if wrong:
                raise RuntimeError(f"non-FP32 detector outputs: {wrong}")
            moved = [
                {"boxes": target["boxes"].to("cuda"), "labels": target["labels"].to("cuda")}
                for target in targets
            ]
            loss, _ = hybrid_loss(loss_fn, outputs, moved)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {step}")
            window = min(args.grad_accum, len(loader) - (step // args.grad_accum) * args.grad_accum)
            (loss / window).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 0.1)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"non-finite gradient norm at epoch {epoch}, batch {step}")
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        history.append({"epoch": epoch, "seconds": time.monotonic() - started})
        torch.save(
            {
                "trainable": trainable_state(wrapper),
                "epoch": epoch,
                "classes": list(CLASSES),
                "history": history,
                "scope": "public SurgHint HeiCo annotations only",
            },
            args.output / "last_head.pth",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
