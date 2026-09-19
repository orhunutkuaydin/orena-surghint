"""Train RF-DETR-L on an eight-class COCO dataset and retain the final EMA weights."""

from __future__ import annotations

import math
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from . import config
from .data import atomic_json, file_digest, read_json

CLASSES = ("Sponge", "Clip", "Specimen Bag", "Silicone Loop", "External Drain", "Needle", "Gallstone", "Specimen")
EPOCHS = 20


def image_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Image filenames must be relative to the COCO directory")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_dataset(root: Path) -> dict:
    coco = read_json(root / "_annotations.coco.json")
    categories = sorted(coco["categories"], key=lambda row: int(row["id"]))
    if [row["name"] for row in categories] != list(CLASSES):
        raise ValueError("COCO category order must match the eight detector classes")
    category_ids = {int(row["id"]) for row in categories}
    if len(category_ids) != len(CLASSES):
        raise ValueError("Duplicate COCO category IDs")
    images = {int(row["id"]): row for row in coco["images"]}
    if not images or len(images) != len(coco["images"]):
        raise ValueError("COCO images must have unique IDs")
    if not coco["annotations"]:
        raise ValueError("The dataset has no annotated objects")
    names = [str(row["file_name"]) for row in images.values()]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate image filenames")
    for image in images.values():
        image_file(root, str(image["file_name"]))
        if image["width"] <= 0 or image["height"] <= 0:
            raise ValueError("Image dimensions must be positive")
    ids = set()
    for annotation in coco["annotations"]:
        if annotation["id"] in ids:
            raise ValueError("Duplicate annotation IDs")
        ids.add(annotation["id"])
        if annotation["image_id"] not in images or annotation["category_id"] not in category_ids:
            raise ValueError("Invalid COCO image or category reference")
        x, y, width, height = annotation["bbox"]
        image = images[annotation["image_id"]]
        if (not all(map(math.isfinite, (x, y, width, height))) or min(x, y) < 0
                or min(width, height) <= 0 or x + width > image["width"] + 1e-4
                or y + height > image["height"] + 1e-4):
            raise ValueError(f"Invalid bounding box: {annotation['id']}")
    return coco


def validation_images(coco: dict, count: int = 64) -> set[int]:
    """Select a deterministic training subset for RF-DETR's required validation loader."""
    if count < len(CLASSES):
        raise ValueError("The subset size must be at least the number of classes")
    images = {int(row["id"]): row for row in coco["images"]}
    by_image = defaultdict(set)
    for annotation in coco["annotations"]:
        by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))
    ordered = sorted(images, key=lambda key: images[key]["file_name"])
    ranks = {key: rank for rank, key in enumerate(ordered)}
    remaining, selected = set(ordered), set()
    uncovered = set().union(*by_image.values())
    while uncovered:
        candidate = max(remaining, key=lambda key: (len(by_image[key] & uncovered), -ranks[key]))
        selected.add(candidate)
        remaining.remove(candidate)
        uncovered -= by_image[candidate]
    for key in ordered:
        if len(selected) >= min(count, len(images)):
            break
        selected.add(key)
    return selected


def main() -> None:
    import torch
    from rfdetr import RFDETRLarge

    if version("rfdetr") != "1.9.1":
        raise RuntimeError("Detector training requires rfdetr==1.9.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select one CUDA GPU for detector training")
    source = config.DETECTOR_DATA.resolve()
    coco = validate_dataset(source)
    pretrained = config.DETECTOR_PRETRAINED.resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    output = config.DETECTOR_OUTPUT.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "dataset"
    dataset.mkdir()
    (dataset / "train").symlink_to(source, target_is_directory=True)
    selected = validation_images(coco)
    subset = {**coco, "images": [row for row in coco["images"] if int(row["id"]) in selected],
              "annotations": [row for row in coco["annotations"] if int(row["image_id"]) in selected]}
    valid = dataset / "valid"
    atomic_json(valid / "_annotations.coco.json", subset)
    for image in subset["images"]:
        destination = valid / image["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(image_file(source, image["file_name"]))
    atomic_json(output / "training_config.json", {
        "images": len(coco["images"]), "classes": list(CLASSES), "epochs": EPOCHS,
        "resolution": 896, "batch_size": 4, "gradient_accumulation_steps": 4,
        "pretrained_sha256": file_digest(pretrained),
        "annotations_sha256": file_digest(source / "_annotations.coco.json"),
        "validation": "Training subset; not a held-out evaluation set",
        "checkpoint_selection": "Final epoch EMA",
    })
    model = RFDETRLarge(resolution=896, device="cuda:0", pretrain_weights=str(pretrained))
    model.train(
        dataset_dir=str(dataset), output_dir=str(output / "training"), resolution=896,
        epochs=EPOCHS, batch_size=4, grad_accum_steps=4, num_workers=config.NUM_WORKERS,
        dataset_file="roboflow", square_resize_div_64=True, multi_scale=False,
        expanded_scales=False, scale_jitter=True, early_stopping=False, use_ema=True,
        eval_ema_only=True, eval_interval=EPOCHS, skip_best_epochs=EPOCHS,
        checkpoint_interval=10, run_test=False, tensorboard=False, wandb=False,
        progress_bar="tqdm", class_names=list(CLASSES), seed=42,
    )
    checkpoint = output / "training/last_ema.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Final EMA checkpoint was not produced: {checkpoint}")
    atomic_json(output / "completed.json", {"checkpoint": "training/last_ema.pth", "sha256": file_digest(checkpoint)})
    print(f"Detector checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
