#!/usr/bin/env python3
"""Prepare FRAME metadata, detector COCO data, and VLM detector evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from surghint.hints import raw_from_detections, render_training_hint
from surghint.postprocessing import CLASSES, RAW_INPUT_FLOOR

EXPECTED_QUESTIONS = 20_000
EXPECTED_TRAIN_QUESTIONS = 13_748
EXPECTED_TEST_QUESTIONS = 6_252
EXPECTED_FRAMES = 15_212
HINT_CONTENT_SHA256 = "51bc17b684fb919ced0207f71cbc55e2c6867f386023c4ad7f225c78fa2c02ee"
FPS = {"heico": 25.0, "lapchole": 30.0}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return [json.loads(item.read_text()) for item in sorted(path.glob("*.json"))]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def seconds(value: str) -> int:
    hours, minutes, secs = map(int, value.split(":"))
    return 3600 * hours + 60 * minutes + secs


def add_frame_columns(frame: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "dataset", dataset)
    frame["split"] = split
    frame["fps"] = FPS[dataset]
    frame["instance_id"] = frame["id"].map(lambda value: f"{dataset}:{value}")
    frame["frame_index"] = (frame["timestamp_start"].map(seconds) * FPS[dataset]).round().astype(int)
    stems = frame["video"].map(lambda value: Path(str(value)).stem)
    names = frame["frame_index"].map(lambda value: f"frame{value:07d}.jpg")
    frame["frame_id"] = [
        f"{dataset}__{stem}__{name[:-4]}" for stem, name in zip(stems, names, strict=True)
    ]
    frame["image_path"] = [
        (Path("frames") / dataset / stem / name).as_posix()
        for stem, name in zip(stems, names, strict=True)
    ]
    for label in ("alt_1", "alt_3"):
        names = frame["frame_index"].map(lambda value: f"frame{value:07d}_{label}.jpg")
        frame[f"image_path_{label}"] = [
            (Path("frames_alt") / dataset / stem / name).as_posix()
            for stem, name in zip(stems, names, strict=True)
        ]
    return frame


def find_parquet(root: Path, split: str) -> Path:
    candidates = [root / f"{split}.parquet", root / "data/frame" / f"{split}.parquet"]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one FRAME {split}.parquet under {root}")
    return found[0]


def find_video(root: Path, name: str) -> Path:
    candidates = [root / "videos" / name, root / name]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one source video {name!r} under {root}")
    return found[0]


def extract_frames(frame: pd.DataFrame, sources: dict[str, Path], output: Path) -> None:
    import decord

    decord.bridge.set_bridge("native")
    unique = frame.drop_duplicates(["dataset", "video", "frame_index"])
    for (dataset, video), group in unique.groupby(["dataset", "video"], sort=True):
        reader = decord.VideoReader(str(find_video(sources[dataset], str(video))), ctx=decord.cpu(0))
        if abs(float(reader.get_avg_fps()) - FPS[dataset]) > 0.1:
            raise ValueError(f"unexpected FPS for {video}: {reader.get_avg_fps()}")
        jobs: dict[int, list[Path]] = defaultdict(list)
        for row in group.itertuples(index=False):
            jobs[max(0, min(int(row.frame_index), len(reader) - 1))].append(output / row.image_path)
            jobs[max(0, min(int(row.frame_index) - 1, len(reader) - 1))].append(
                output / row.image_path_alt_1
            )
            jobs[max(0, min(int(row.frame_index) + 1, len(reader) - 1))].append(
                output / row.image_path_alt_3
            )
        indexes = sorted(jobs)
        for start in range(0, len(indexes), 32):
            batch_indexes = indexes[start : start + 32]
            arrays = reader.get_batch(batch_indexes).asnumpy()
            for index, array in zip(batch_indexes, arrays, strict=True):
                for path in jobs[index]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array, mode="RGB").save(
                        path, format="JPEG", quality=95, subsampling=0
                    )


def metadata(args: argparse.Namespace) -> None:
    sources = {"heico": args.heico.resolve(), "lapchole": args.lapchole.resolve()}
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    for dataset, root in sources.items():
        for split in ("train", "test"):
            parts[split].append(add_frame_columns(pd.read_parquet(find_parquet(root, split)), dataset, split))
    frames = {split: pd.concat(value, ignore_index=True) for split, value in parts.items()}
    if (len(frames["train"]), len(frames["test"])) != (
        EXPECTED_TRAIN_QUESTIONS,
        EXPECTED_TEST_QUESTIONS,
    ):
        raise ValueError(
            f"question counts are {len(frames['train'])}/{len(frames['test'])}, "
            "expected 13748/6252"
        )
    combined = pd.concat(frames.values(), ignore_index=True)
    if len(combined) != EXPECTED_QUESTIONS or combined["instance_id"].nunique() != EXPECTED_QUESTIONS:
        raise ValueError("combined metadata is not exactly 20,000 unique questions")
    if combined["frame_id"].nunique() != EXPECTED_FRAMES:
        raise ValueError("combined metadata is not exactly 15,212 unique frames")
    if set(frames["train"]["frame_id"]) & set(frames["test"]["frame_id"]):
        raise ValueError("train/test frame overlap")
    if args.extract:
        extract_frames(combined, sources, args.output)
    missing = [
        path for path in combined["image_path"].drop_duplicates() if not (args.output / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} center frame(s), first: {missing[0]}")
    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.sort_values(["dataset", "video", "id"], kind="stable").to_parquet(
            metadata_dir / f"{split}.parquet", index=False
        )
    print(f"PASS: 20,000 questions, 15,212 frames -> {args.output}")


def source_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = str(row.get("frame_id") or row.get("metadata_id") or row.get("id") or "")
        if not key or key in result:
            raise ValueError("source map keys are missing or duplicated")
        result[key] = row
    return result


def box_xyxy(item: dict[str, Any]) -> list[float]:
    value = item.get("bbox_xyxy") or item.get("xyxy")
    if value is None and item.get("bbox") is not None:
        x, y, width, height = map(float, item["bbox"])
        value = [x, y, x + width, y + height]
    if value is None or len(value) != 4:
        raise ValueError(f"object has no four-value bounding box: {item}")
    return [float(number) for number in value]


def mapped_video(root: Path, value: str) -> Path:
    relative = Path(value)
    candidates = {
        root / relative,
        root / relative.name,
        root / "videos" / relative.name,
        root / "heico" / relative.name,
    }
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one source video for {value!r} under {root}")
    return found[0]


def annotation_coco(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.annotations.resolve())
    mapping = source_map(args.source_map.resolve() if args.source_map else None)
    if bool(args.images) == bool(args.videos):
        raise ValueError("provide exactly one of --images or --videos")
    readers = {}
    if args.videos:
        import decord

        decord.bridge.set_bridge("native")
    images, annotations = [], []
    for image_id, row in enumerate(rows, 1):
        frame_id = str(row.get("frame_id") or row.get("metadata_id") or row.get("id") or "")
        mapped = mapping.get(frame_id, {})
        relative = str(
            row.get("image_path")
            or row.get("source_image_path")
            or mapped.get("image_path")
            or mapped.get("source_image_path")
            or mapped.get("source_image_file")
            or ""
        )
        if not frame_id or not relative:
            raise ValueError(f"annotation row lacks frame/source mapping: {row}")
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"heico__{frame_id}")
        filename = str(Path(filename).with_suffix(Path(relative).suffix.lower() or ".jpg"))
        destination = args.output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.images:
            source = (args.images / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as handle:
                width, height = handle.size
            if not destination.exists():
                destination.symlink_to(source)
        else:
            video_name = str(mapped.get("source_video_file") or "")
            match = re.search(r"frame(\d+)", Path(relative).name)
            if not video_name or not match:
                raise ValueError(f"source mapping lacks video/frame index: {mapped}")
            video = mapped_video(args.videos.resolve(), video_name)
            reader = readers.get(video)
            if reader is None:
                reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
                readers[video] = reader
            frame_index = int(match.group(1))
            if frame_index >= len(reader):
                raise IndexError(f"frame {frame_index} outside {video}")
            if not destination.exists():
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                Image.fromarray(reader[frame_index].asnumpy(), mode="RGB").save(
                    temporary, format="JPEG", quality=95, subsampling=0
                )
                os.replace(temporary, destination)
            with Image.open(destination) as handle:
                width, height = handle.size
        declared = (int(row.get("width", width)), int(row.get("height", height)))
        if (width, height) != declared:
            raise ValueError(f"dimension mismatch for {frame_id}: {(width, height)} != {declared}")
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "frame_id": frame_id,
                "width": width,
                "height": height,
                "source_kind": "surghint_heico",
                "allow_empty": True,
            }
        )
        for item in row.get("objects") or row.get("annotations") or []:
            name = str(item.get("fo_class") or item.get("class_name") or item.get("label") or "")
            if name not in CLASSES:
                continue
            x1, y1, x2, y2 = box_xyxy(item)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CLASSES.index(name) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": max(0, x2 - x1) * max(0, y2 - y1),
                    "iscrowd": 0,
                }
            )
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "foreign_object"}
            for index, name in enumerate(CLASSES)
        ],
    }
    write_json(args.output / "_annotations.coco.json", payload)
    print(f"PASS: {len(images)} images, {len(annotations)} boxes -> {args.output}")


def hint_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("frame_id").to_dict("records"):
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
        )
    return digest.hexdigest()


def hints(args: argparse.Namespace) -> None:
    boxes = pd.read_parquet(args.boxes)
    manifest = pd.read_parquet(args.manifest)
    dimensions = {}
    if args.dimensions:
        dimension_frame = pd.read_parquet(args.dimensions)
        if not {"frame_id", "width", "height"} <= set(dimension_frame):
            raise ValueError("dimension parquet lacks frame_id/width/height")
        dimensions = {
            str(row.frame_id): (int(row.width), int(row.height))
            for row in dimension_frame.itertuples(index=False)
        }
    required = {
        "frame_id", "dataset", "video", "split", "fold", "source", "threshold",
        "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height",
    }
    if not required <= set(boxes):
        raise ValueError(f"raw detector boxes lack columns: {sorted(required - set(boxes))}")
    if len(manifest) != EXPECTED_FRAMES or manifest["frame_id"].astype(str).nunique() != EXPECTED_FRAMES:
        raise ValueError("frame manifest must contain 15,212 unique frames")
    if set(boxes["frame_id"].astype(str)) - set(manifest["frame_id"].astype(str)):
        raise ValueError("raw boxes contain frames outside the manifest")
    if set(boxes["fold"].astype(int)) - {0, 1, 2, 3}:
        raise ValueError("raw boxes are not four-fold out-of-fold predictions")
    if not np.allclose(boxes["threshold"].astype(float), RAW_INPUT_FLOOR):
        raise ValueError("raw box floor is not 0.05")
    grouped = {str(key): value for key, value in boxes.groupby("frame_id", sort=False)}
    records = []
    for row in manifest.sort_values("frame_id").itertuples(index=False):
        group = grouped.get(str(row.frame_id))
        if group is None:
            if hasattr(row, "width") and hasattr(row, "height"):
                width, height = int(row.width), int(row.height)
            elif str(row.frame_id) in dimensions:
                width, height = dimensions[str(row.frame_id)]
            elif hasattr(row, "image_path") and Path(str(row.image_path)).is_file():
                with Image.open(str(row.image_path)) as image:
                    width, height = image.size
            else:
                raise ValueError(f"no dimensions for empty-detection frame {row.frame_id}")
            detections, fold = [], int(row.fold)
        else:
            first = group.iloc[0]
            width, height, fold = int(first.width), int(first.height), int(first.fold)
            detections = [
                (
                    str(item.class_name), float(item.confidence), float(item.x1), float(item.y1),
                    float(item.x2), float(item.y2),
                )
                for item in group.itertuples(index=False)
            ]
        raw = raw_from_detections(detections)
        rendered = render_training_hint(raw, width=width, height=height)
        records.append(
            {
                "frame_id": str(row.frame_id),
                "dataset": str(row.dataset),
                "video": str(row.video),
                "split": str(row.split),
                "fold": fold,
                "source": "dinov3_vit7b16_out_of_fold",
                "threshold": 0.30,
                "external_drain_threshold": 0.45,
                "n_boxes": 0,
                "n_candidates": rendered.count(" candidate:"),
                "width": width,
                "height": height,
                "hint_format": "detector_boxes_with_candidates",
                "hint": rendered,
            }
        )
    output = pd.DataFrame(records)
    # Parse counts from the rendered evidence to keep one selection implementation.
    output["n_boxes"] = output["hint"].map(
        lambda text: sum(
            int(match.group(1))
            for match in re.finditer(
                r"^(?:Sponge|Clip|Specimen bag|Silicone loop|External drain|Needle|Gallstone|Specimen): (\d+)$",
                text,
                re.MULTILINE,
            )
        )
    )
    digest = hint_digest(output)
    if args.strict and digest != HINT_CONTENT_SHA256:
        raise ValueError(f"hint content sha256 {digest}, expected {HINT_CONTENT_SHA256}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    (args.output.with_suffix(args.output.suffix + ".content.sha256")).write_text(digest + "\n")
    print(f"PASS: {len(output)} hints, content sha256 {digest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    meta = commands.add_parser("metadata")
    meta.add_argument("--heico", required=True, type=Path)
    meta.add_argument("--lapchole", required=True, type=Path)
    meta.add_argument("--output", required=True, type=Path)
    meta.add_argument("--extract", action="store_true")
    coco = commands.add_parser("coco")
    coco.add_argument("--annotations", required=True, type=Path)
    coco.add_argument("--source-map", type=Path)
    coco.add_argument("--images", type=Path)
    coco.add_argument("--videos", type=Path)
    coco.add_argument("--output", required=True, type=Path)
    render = commands.add_parser("hints")
    render.add_argument("--boxes", required=True, type=Path)
    render.add_argument("--manifest", required=True, type=Path)
    render.add_argument("--dimensions", type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    {"metadata": metadata, "coco": annotation_coco, "hints": hints}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
