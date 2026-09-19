"""Convert Plain-DETR detections into auxiliary evidence for Qwen3.5-9B."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .hint_formatting import (
    CANDIDATE_HEADER,
    HEADER,
    SUMMARY_HEADER,
    HintBox,
    render_hint as format_hint,
)
from .postprocessing import (
    CLASS_INDEX,
    CLASSES,
    RAW_INPUT_FLOOR,
    SelectionPolicy,
    filter_candidate_classes,
    filter_cross_class_candidates,
    filter_enclosing_candidates,
    select_aggregated_candidates,
    select_confirmed_boxes,
    select_confirmed_boxes_for_candidates,
)

_EMPTY = np.empty((0, 6), dtype=float)


def raw_from_detections(
    detections: Sequence[tuple[str, float, float, float, float, float]],
) -> np.ndarray:
    """Detector output rows -> the (N, 6) raw array the pipeline consumes.

    Rows are ``(class_name, confidence, x1, y1, x2, y2)`` in original-image
    pixels with confidence already floored at ``RAW_INPUT_FLOOR`` (0.05).
    """

    if not detections:
        return _EMPTY
    rows = []
    for name, confidence, x1, y1, x2, y2 in detections:
        if name not in CLASS_INDEX:
            raise ValueError(f"unknown detector class: {name!r}")
        if float(confidence) + 1e-9 < RAW_INPUT_FLOOR:
            raise ValueError(
                f"box below the raw floor {RAW_INPUT_FLOOR}: {confidence!r}"
            )
        rows.append(
            (
                float(CLASS_INDEX[name]),
                float(confidence),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )
        )
    return np.asarray(rows, dtype=float)


def _to_hint_box(row: np.ndarray) -> HintBox:
    return HintBox(
        CLASSES[int(row[0])],
        float(row[1]),
        *(float(value) for value in row[2:6]),
    )


def render_hint(raw: np.ndarray, *, width: int, height: int) -> str:
    """Render detector evidence for one frame."""

    if raw.ndim != 2 or (len(raw) and raw.shape[1] != 6):
        raise ValueError(f"raw boxes must be (N, 6), got {raw.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame dimensions {width}x{height}")

    confirmed, _ = select_confirmed_boxes_for_candidates(raw)
    candidates = select_aggregated_candidates(raw, confirmed)
    candidates, _ = filter_candidate_classes(candidates, confirmed)
    candidates, _ = filter_cross_class_candidates(candidates, confirmed)
    confirmed, _ = select_confirmed_boxes(raw)
    candidates, _ = filter_enclosing_candidates(
        candidates, confirmed, width=width, height=height
    )
    if len({item["class_name"] for item in candidates}) != len(candidates):
        raise RuntimeError("multiple candidates per class")

    rendered = format_hint(
        [_to_hint_box(row) for row in confirmed],
        [
            HintBox(
                str(item["class_name"]),
                float(item["confidence"]),
                *(float(value) for value in item["box"]),
            )
            for item in candidates
        ],
        width=width,
        height=height,
    )
    if not rendered.startswith(HEADER):
        raise RuntimeError("wrong hint header")
    if rendered.count(CANDIDATE_HEADER) != 1:
        raise RuntimeError("wrong candidate header count")
    if rendered.rfind(SUMMARY_HEADER) < rendered.find(CANDIDATE_HEADER):
        raise RuntimeError("summary is not last")
    if rendered.endswith("\n") or "\n\n" in rendered or not rendered.isascii():
        raise RuntimeError("hint byte-format violation")
    return rendered


def render_training_hint(raw: np.ndarray, *, width: int, height: int) -> str:
    """Render the detector evidence used for LoRA training."""

    if raw.ndim != 2 or (len(raw) and raw.shape[1] != 6):
        raise ValueError(f"raw boxes must be (N, 6), got {raw.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame dimensions {width}x{height}")
    confirmed, _ = select_confirmed_boxes(raw)
    candidates = select_aggregated_candidates(raw, confirmed)
    candidates, _ = filter_candidate_classes(candidates, confirmed)
    candidates, _ = filter_cross_class_candidates(candidates, confirmed)
    candidates, _ = filter_enclosing_candidates(
        candidates, confirmed, width=width, height=height
    )
    rendered = format_hint(
        [_to_hint_box(row) for row in confirmed],
        [
            HintBox(
                str(item["class_name"]),
                float(item["confidence"]),
                *(float(value) for value in item["box"]),
            )
            for item in candidates
        ],
        width=width,
        height=height,
    )
    if (
        not rendered.startswith(HEADER)
        or rendered.count(CANDIDATE_HEADER) != 1
        or rendered.rfind(SUMMARY_HEADER) < rendered.find(CANDIDATE_HEADER)
        or rendered.endswith("\n")
        or "\n\n" in rendered
        or not rendered.isascii()
    ):
        raise RuntimeError("hint byte-format violation")
    return rendered


# Counting questions use Gaussian soft-NMS before the standard selection rules.

SOFTNMS_SIGMA = 0.5
SOFTNMS_FLOOR = 0.10
SOFTNMS_POLICY = SelectionPolicy(0.25, 0.50, 0.15, 0.50, 3)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 2], b[None, :, 2])
    iy1 = np.maximum(a[:, None, 3], b[None, :, 3])
    ix2 = np.minimum(a[:, None, 4], b[None, :, 4])
    iy2 = np.minimum(a[:, None, 5], b[None, :, 5])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 4] - a[:, 2]) * (a[:, 5] - a[:, 3])
    area_b = (b[:, 4] - b[:, 2]) * (b[:, 5] - b[:, 3])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def soft_nms(raw: np.ndarray) -> np.ndarray:
    """(N, 6) raw rows -> (M, 7) survivors: decayed confidence in column 1, original in column 6."""

    if not len(raw):
        return np.empty((0, 7), dtype=float)
    rows7 = np.concatenate([raw[:, :6], raw[:, 1:2]], axis=1).astype(float)
    out = []
    for class_index in np.unique(rows7[:, 0]):
        rows = rows7[rows7[:, 0] == class_index].copy()
        while len(rows):
            best_index = int(np.argmax(rows[:, 1]))
            best = rows[best_index].copy()
            out.append(best)
            rows = np.delete(rows, best_index, axis=0)
            if not len(rows):
                break
            rows[:, 1] *= np.exp(-(_iou_matrix(best[None], rows)[0] ** 2) / SOFTNMS_SIGMA)
            rows = rows[rows[:, 1] >= SOFTNMS_FLOOR]
    return np.array(out, dtype=float) if out else np.empty((0, 7), dtype=float)


def render_counting_hint(raw: np.ndarray, *, width: int, height: int) -> str:
    """Render detector evidence for counting questions."""

    if raw.ndim != 2 or (len(raw) and raw.shape[1] != 6):
        raise ValueError(f"raw boxes must be (N, 6), got {raw.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame dimensions {width}x{height}")
    survivors = soft_nms(raw)
    decayed = survivors[:, :6] if len(survivors) else np.empty((0, 6), dtype=float)
    confirmed, _ = select_confirmed_boxes_for_candidates(decayed, SOFTNMS_POLICY)
    candidates = select_aggregated_candidates(decayed, confirmed, SOFTNMS_POLICY)
    candidates, _ = filter_candidate_classes(candidates, confirmed)
    candidates, _ = filter_cross_class_candidates(candidates, confirmed)
    confirmed, _ = select_confirmed_boxes(decayed, SOFTNMS_POLICY)
    candidates, _ = filter_enclosing_candidates(
        candidates, confirmed, width=width, height=height
    )
    if len({item["class_name"] for item in candidates}) != len(candidates):
        raise RuntimeError("multiple candidates per class")
    original = {tuple(np.round(row[2:6], 4)): float(row[6]) for row in survivors}

    def original_confidence(row: np.ndarray) -> float:
        return original.get(tuple(np.round(row[2:6], 4)), float(row[1]))

    rendered = format_hint(
        [
            HintBox(CLASSES[int(row[0])], original_confidence(row), *(float(value) for value in row[2:6]))
            for row in confirmed
        ],
        [
            HintBox(
                str(item["class_name"]),
                float(item["confidence"]),
                *(float(value) for value in item["box"]),
            )
            for item in candidates
        ],
        width=width,
        height=height,
    )
    if not rendered.startswith(HEADER):
        raise RuntimeError("wrong hint header")
    if rendered.count(CANDIDATE_HEADER) != 1:
        raise RuntimeError("wrong candidate header count")
    if rendered.rfind(SUMMARY_HEADER) < rendered.find(CANDIDATE_HEADER):
        raise RuntimeError("summary is not last")
    if rendered.endswith("\n") or "\n\n" in rendered or not rendered.isascii():
        raise RuntimeError("hint byte-format violation")
    return rendered
