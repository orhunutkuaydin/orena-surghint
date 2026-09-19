"""Class-aware box filtering and timestamped RGB frame rendering."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

CLASSES = (
    "Sponge",
    "Clip",
    "Specimen Bag",
    "Silicone Loop",
    "External Drain",
    "Needle",
    "Gallstone",
    "Specimen",
    "Mesh",
    "Absorbable Hemostatic Agent",
)


TAGS = ("SP", "CL", "SB", "SL", "ED", "NE", "GA", "SE", "ME", "HA")


COLORS = (
    (40, 220, 255),
    (255, 180, 60),
    (90, 240, 90),
    (240, 80, 220),
    (60, 140, 255),
    (255, 255, 80),
    (160, 120, 255),
    (255, 160, 200),
    (100, 220, 180),
    (220, 220, 220),
)


@dataclass(frozen=True)
class EvidenceConfig:
    threshold: float = 0.5
    nms_iou: float = 0.5
    canvas_width: int = 768
    canvas_height: int = 448
    header_height: int = 32
    box_thickness: int = 1
    detector_classes: tuple[str, ...] = CLASSES[:8]


def timestamp(seconds: float) -> str:
    """Format absolute procedure time as hh:mm:ss.s."""
    ticks = max(0, int(math.floor(float(seconds) * 10 + 0.5)))
    hours, rest = divmod(ticks, 36000)
    minutes, rest = divmod(rest, 600)
    whole_seconds, fraction = divmod(rest, 10)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction}"


def filter_detections(
    boxes: Any,
    confidence: Any,
    class_ids: Any,
    width: int,
    height: int,
    cfg: EvidenceConfig,
) -> dict[str, list]:
    """Apply class-wise NMS and return normalized source-frame boxes."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(confidence, dtype=np.float64).reshape(-1)
    c = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if len(b) != len(s) or len(b) != len(c) or min(width, height) <= 0:
        raise ValueError("Inconsistent RF-DETR outputs")
    if np.any((c < 0) | (c >= len(cfg.detector_classes))):
        raise ValueError("Checkpoint class indices disagree with DETECTOR_CLASSES")
    keep = np.isfinite(b).all(axis=1) & np.isfinite(s) & (s >= cfg.threshold)
    b, s, c = (b[keep], s[keep], c[keep])
    b[:, [0, 2]] = np.clip(b[:, [0, 2]] / width, 0, 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]] / height, 0, 1)
    valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b, s, c = (b[valid], s[valid], c[valid])
    selected = []
    for class_id in sorted(set(c.tolist())):
        indices = np.flatnonzero(c == class_id)
        indices = indices[np.argsort(-s[indices], kind="stable")]
        while len(indices):
            i = int(indices[0])
            selected.append(i)
            rest = indices[1:]
            lt = np.maximum(b[i, :2], b[rest, :2])
            rb = np.minimum(b[i, 2:], b[rest, 2:])
            inter = np.prod(np.maximum(rb - lt, 0), axis=1)
            area_i = np.prod(b[i, 2:] - b[i, :2])
            area_r = np.prod(b[rest, 2:] - b[rest, :2], axis=1)
            iou = inter / np.maximum(area_i + area_r - inter, 1e-12)
            indices = rest[iou <= cfg.nms_iou]
    selected.sort(key=lambda i: (int(c[i]), -float(s[i]), float(b[i, 0])))
    return {
        "boxes": b[selected].tolist(),
        "scores": s[selected].tolist(),
        "classes": c[selected].tolist(),
    }


def render_frame(
    rgb: np.ndarray,
    detection: dict,
    absolute_time: float,
    frame_number: int,
    cfg: EvidenceConfig,
) -> np.ndarray:
    """Render boxes and a timestamp on an aspect-ratio-preserving canvas."""
    height, width = rgb.shape[:2]
    available_h = cfg.canvas_height - cfg.header_height
    scale = min(cfg.canvas_width / width, available_h / height)
    w, h = (max(1, round(width * scale)), max(1, round(height * scale)))
    ox, oy = ((cfg.canvas_width - w) // 2, cfg.header_height + (available_h - h) // 2)
    frame = np.zeros((cfg.canvas_height, cfg.canvas_width, 3), dtype=np.uint8)
    resized = cv2.resize(
        rgb, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )
    frame[oy : oy + h, ox : ox + w] = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.putText(
        frame,
        f"F{frame_number:02d}  absolute {timestamp(absolute_time)}",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for box, score, c in zip(
        detection["boxes"], detection["scores"], detection["classes"], strict=True
    ):
        global_c = CLASSES.index(cfg.detector_classes[c])
        color = COLORS[global_c]
        x1 = int(np.clip(ox + round(box[0] * w), ox, ox + w - 1))
        y1 = int(np.clip(oy + round(box[1] * h), oy, oy + h - 1))
        x2 = int(np.clip(ox + round(box[2] * w), x1, ox + w - 1))
        y2 = int(np.clip(oy + round(box[3] * h), y1, oy + h - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.box_thickness)
        label = f"{TAGS[global_c]} {score:.2f}"
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        tx = max(0, min(x1, cfg.canvas_width - label_w - 2))
        ty = y1 - 4 if y1 - 4 > cfg.header_height + 10 else min(oy + h - 2, y2 + 13)
        cv2.putText(
            frame,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_video(
    frames: np.ndarray, detections: list[dict], times: list[float], cfg: EvidenceConfig
) -> np.ndarray:
    if not len(frames) == len(detections) == len(times):
        raise ValueError("Selected pixels, boxes and timestamps must align exactly")
    return np.stack(
        [
            render_frame(rgb, det, t, i + 1, cfg)
            for i, (rgb, det, t) in enumerate(
                zip(frames, detections, times, strict=True)
            )
        ]
    )
