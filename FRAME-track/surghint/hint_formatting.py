"""Render deterministic detector evidence for the FRAME vision-language model."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

HINT_CLASSES: tuple[str, ...] = (
    "Sponge",
    "Clip",
    "Specimen bag",
    "Silicone loop",
    "External drain",
    "Needle",
    "Gallstone",
    "Specimen",
)

DETECTOR_CLASS_TO_HINT: dict[str, str] = {
    "Sponge": "Sponge",
    "Clip": "Clip",
    "Specimen Bag": "Specimen bag",
    "Silicone Loop": "Silicone loop",
    "External Drain": "External drain",
    "Needle": "Needle",
    "Gallstone": "Gallstone",
    "Specimen": "Specimen",
}

# The wording is part of the model input and must remain stable.
HEADER = (
    "Auxiliary box detections (per class count, then each box: "
    "confidence, position, distance from image center):"
)
CANDIDATE_HEADER = "Unconfirmed candidate boxes (verify in image; not included in summary):"
SUMMARY_HEADER = "Box summary:"
MAX_INSTANCE_LINES = 10


@dataclass(frozen=True)
class HintBox:
    """One detector box in source-image pixels."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


def quadrant_of(center_x: float, center_y: float, width: float, height: float) -> str:
    vertical = "top" if center_y < height / 2.0 else "bottom"
    horizontal = "left" if center_x < width / 2.0 else "right"
    return f"{vertical}/{horizontal}"


def center_distance(center_x: float, center_y: float, width: float, height: float) -> float:
    half_diagonal = math.hypot(width, height) / 2.0
    if half_diagonal <= 0:
        return 0.0
    distance = math.hypot(center_x - width / 2.0, center_y - height / 2.0)
    return min(1.0, distance / half_diagonal)


def _hint_name(box: HintBox) -> str:
    if box.class_name in HINT_CLASSES:
        return box.class_name
    if box.class_name not in DETECTOR_CLASS_TO_HINT:
        raise ValueError(f"unknown detector class: {box.class_name!r}")
    return DETECTOR_CLASS_TO_HINT[box.class_name]


def _render_confirmed(
    boxes: Iterable[HintBox],
    *,
    width: float,
    height: float,
) -> str:
    per_class: dict[str, list[HintBox]] = {name: [] for name in HINT_CLASSES}
    for box in boxes:
        per_class[_hint_name(box)].append(box)

    lines = [HEADER]
    total = 0
    closest_key: tuple[float, int, float] | None = None
    closest_name = "none"
    for class_index, name in enumerate(HINT_CLASSES):
        instances = sorted(
            per_class[name],
            key=lambda box: (-box.confidence, box.x1, box.y1, box.x2, box.y2),
        )
        total += len(instances)
        lines.append(f"{name}: {len(instances)}")
        for instance_number, box in enumerate(instances[:MAX_INSTANCE_LINES], 1):
            center_x, center_y = box.center()
            distance = center_distance(center_x, center_y, width, height)
            lines.append(
                f"{name} {instance_number}: {box.confidence:.2f} "
                f"{quadrant_of(center_x, center_y, width, height)} d={distance:.2f}"
            )
        for box in instances:
            center_x, center_y = box.center()
            key = (
                center_distance(center_x, center_y, width, height),
                class_index,
                -box.confidence,
            )
            if closest_key is None or key < closest_key:
                closest_key, closest_name = key, name

    visible = sorted(name for name in HINT_CLASSES if per_class[name])
    lines.append(SUMMARY_HEADER)
    lines.append(f"Visible = {', '.join(visible) if visible else 'none'}")
    lines.append(f"Distinct classes = {len(visible)}")
    lines.append(f"Total instances = {total}")
    lines.append(f"Closest to image center = {closest_name}")
    return "\n".join(lines)


def render_hint(
    confirmed: Sequence[HintBox],
    candidates: Sequence[HintBox],
    *,
    width: float,
    height: float,
) -> str:
    """Render confirmed detections, candidate evidence, and summary."""

    per_class: dict[str, HintBox] = {}
    for box in candidates:
        name = _hint_name(box)
        if name in per_class:
            raise ValueError(f"multiple candidates for {name}")
        per_class[name] = box

    candidate_lines = [CANDIDATE_HEADER]
    for name in HINT_CLASSES:
        box = per_class.get(name)
        if box is None:
            continue
        center_x, center_y = box.center()
        candidate_lines.append(
            f"{name} candidate: {box.confidence:.2f} "
            f"{quadrant_of(center_x, center_y, width, height)} "
            f"d={center_distance(center_x, center_y, width, height):.2f}"
        )
    if len(candidate_lines) == 1:
        candidate_lines.append("none")

    base = _render_confirmed(confirmed, width=width, height=height)
    marker = f"\n{SUMMARY_HEADER}\n"
    if marker not in base:
        raise RuntimeError("summary marker is absent")
    detections, summary = base.split(marker, 1)
    return detections + "\n" + "\n".join(candidate_lines) + marker + summary
