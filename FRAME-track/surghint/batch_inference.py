#!/usr/bin/env python3
"""Run FRAME inference for rows from prepared metadata."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .inference import answers
from .weights import prepare_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    return parser.parse_args()


def read_metadata(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("--metadata must be a .parquet or .csv file")


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        handle.verify()
    with Image.open(path) as handle:
        return handle.convert("RGB")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")

    metadata = read_metadata(args.metadata.resolve())
    required = {"image_path", "question"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"metadata lacks columns: {sorted(missing_columns)}")
    selected = metadata.head(args.limit).copy()
    if len(selected) != args.limit:
        raise ValueError(f"requested {args.limit} rows, metadata has {len(selected)}")

    data_root = args.data_root.resolve()
    image_paths = [data_root / str(value) for value in selected["image_path"]]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"missing {len(missing_images)} image(s), first: {missing_images[0]}"
        )
    images = [load_image(path) for path in image_paths]
    questions = selected["question"].astype(str).tolist()

    weights = args.weights_dir.resolve()
    prepare_for_inference(weights)
    selected["prediction"] = answers(
        list(zip(images, questions, strict=True)), weights, "cuda"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)

    columns = [
        name
        for name in ("instance_id", "image_path", "question", "answer", "prediction")
        if name in selected.columns
    ]
    print(selected[columns].to_string(index=False))
    print(f"Wrote {len(selected)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
