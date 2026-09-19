#!/usr/bin/env python3
"""Postprocess DINOv3 detections into stable class, count, and location evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CLASSES: tuple[str, ...] = (
    "Sponge",
    "Clip",
    "Specimen Bag",
    "Silicone Loop",
    "External Drain",
    "Needle",
    "Gallstone",
    "Specimen",
)
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
QUADRANTS = ("top/left", "top/right", "bottom/left", "bottom/right")


@dataclass(frozen=True)
class SelectionPolicy:
    threshold: float
    nms_iou: float
    candidate_floor: float
    minimum_iou: float
    maximum_votes: int


UNIFORM_POLICY = SelectionPolicy(0.30, 0.50, 0.15, 0.50, 3)
EXTERNAL_DRAIN_THRESHOLD = 0.45
RAW_INPUT_FLOOR = 0.05
CANDIDATE_ACTIVATION_FLOOR = 0.10
CANDIDATE_SUPPORT_FLOOR = 0.05
MUTUALLY_EXCLUSIVE_PAIR = frozenset(("Specimen Bag", "Silicone Loop"))
MUTUALLY_EXCLUSIVE_IOU = 0.80
CROSS_CLASS_DUPLICATE_IOU = 1.00
CLIP_CONTAINMENT_THRESHOLD = 0.90
CLIP_MINIMUM_AREA_RATIO = 0.35
SAME_CLASS_CONTAINMENT_THRESHOLD = 0.90
SAME_CLASS_MINIMUM_AREA_RATIO = 0.0
SAME_CLASS_PARENT_ONLY_CLASSES = frozenset(name for name in CLASSES if name != "Clip")
SINGLETON_CONFIRMED_CLASSES = frozenset(("Specimen Bag", "Specimen"))
FALLBACK_ONLY_CANDIDATE_CLASSES = frozenset(("Specimen",))
DISABLED_CANDIDATE_CLASSES = frozenset(("Gallstone",))
SAME_CLASS_CANDIDATE_COVERAGE_THRESHOLD = 0.90
CANDIDATE_CROSS_CLASS_DUPLICATE_IOU = 0.99
NEEDLE_SINGLETON_CLASSES = frozenset(("Needle",))
SEMANTIC_ENCLOSING_CANDIDATE_CLASSES = frozenset(
    name for name in CLASSES if name != "Clip"
)


def overlap_metrics(one: np.ndarray, other: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = max(one[0], other[0]), max(one[1], other[1])
    x2, y2 = min(one[2], other[2]), min(one[3], other[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_other = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    union = area_one + area_other - intersection
    smaller, larger = min(area_one, area_other), max(area_one, area_other)
    iou = intersection / union if union else 0.0
    containment = intersection / smaller if smaller else 0.0
    area_ratio = smaller / larger if larger else 0.0
    center_one = ((one[0] + one[2]) / 2, (one[1] + one[3]) / 2)
    center_other = ((other[0] + other[2]) / 2, (other[1] + other[3]) / 2)
    center_gap = math.hypot(center_one[0] - center_other[0], center_one[1] - center_other[1]) / max(
        math.sqrt(larger), 1e-9
    )
    return iou, containment, area_ratio, center_gap


def classwise_nms(rows: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy class-wise NMS over ``class, score, x1, y1, x2, y2`` rows."""

    kept: list[np.ndarray] = []
    for class_index in range(len(CLASSES)):
        members = rows[rows[:, 0] == class_index]
        members = members[np.argsort(-members[:, 1])]
        chosen: list[np.ndarray] = []
        for member in members:
            if any(overlap_metrics(member[2:6], prior[2:6])[0] >= threshold for prior in chosen):
                continue
            chosen.append(member)
        kept.extend(chosen)
    return np.asarray(kept, dtype=float).reshape(-1, 6)


def aligned(first: np.ndarray, second: np.ndarray, minimum_iou: float) -> bool:
    iou, containment, area_ratio, center_gap = overlap_metrics(first, second)
    return iou >= minimum_iou or (containment >= 0.80 and area_ratio >= 0.25 and center_gap <= 0.45)


def unique_votes(members: np.ndarray, maximum_votes: int) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for member in members[np.argsort(-members[:, 1])]:
        if any(overlap_metrics(member[2:6], prior[2:6])[0] >= 0.90 for prior in chosen):
            continue
        chosen.append(member)
        if len(chosen) >= maximum_votes:
            break
    return np.asarray(chosen, dtype=float).reshape(-1, 6)


def suppress_mutually_exclusive_pair(
    selected: np.ndarray,
    reasons: list[str],
    *,
    iou_threshold: float = MUTUALLY_EXCLUSIVE_IOU,
) -> tuple[np.ndarray, list[str]]:
    """Resolve near-identical Specimen Bag/Silicone Loop labels by confidence."""

    if not len(selected):
        return selected, reasons
    pair_indexes = {CLASS_INDEX[name] for name in MUTUALLY_EXCLUSIVE_PAIR}
    order = np.argsort(-selected[:, 1], kind="stable")
    kept_indexes: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        class_index = int(selected[index, 0])
        suppress = False
        if class_index in pair_indexes:
            for prior_index in kept_indexes:
                prior_class = int(selected[prior_index, 0])
                if prior_class not in pair_indexes or prior_class == class_index:
                    continue
                if (
                    overlap_metrics(selected[index, 2:6], selected[prior_index, 2:6])[0]
                    >= iou_threshold
                ):
                    suppress = True
                    break
        if not suppress:
            kept_indexes.append(index)
    return selected[kept_indexes], [reasons[index] for index in kept_indexes]


def suppress_cross_class_duplicate_queries(
    selected: np.ndarray,
    reasons: list[str],
    *,
    iou_threshold: float = CROSS_CLASS_DUPLICATE_IOU,
) -> tuple[np.ndarray, list[str]]:
    """Keep one label when effectively identical geometry has multiple classes."""

    if not len(selected):
        return selected, reasons
    order = np.argsort(-selected[:, 1], kind="stable")
    kept_indexes: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        suppress = any(
            int(selected[index, 0]) != int(selected[prior_index, 0])
            and overlap_metrics(selected[index, 2:6], selected[prior_index, 2:6])[0]
            >= iou_threshold
            for prior_index in kept_indexes
        )
        if not suppress:
            kept_indexes.append(index)
    return selected[kept_indexes], [reasons[index] for index in kept_indexes]


def suppress_contained_clip_duplicates(
    selected: np.ndarray,
    reasons: list[str],
    *,
    containment_threshold: float = CLIP_CONTAINMENT_THRESHOLD,
    minimum_area_ratio: float = CLIP_MINIMUM_AREA_RATIO,
) -> tuple[np.ndarray, list[str]]:
    """Remove nested duplicate Clip boxes missed by ordinary IoU NMS.

    A smaller box can be wholly inside a larger prediction while their IoU is
    below 0.50.  Only Clip receives this extra rule, and a tiny inner proposal
    is preserved unless it covers at least 35% of the stronger box's area.
    """

    if not len(selected):
        return selected, reasons
    clip_index = CLASS_INDEX["Clip"]
    order = np.argsort(-selected[:, 1], kind="stable")
    kept_indexes: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        suppress = False
        if int(selected[index, 0]) == clip_index:
            for prior_index in kept_indexes:
                if int(selected[prior_index, 0]) != clip_index:
                    continue
                _, containment, area_ratio, _ = overlap_metrics(
                    selected[index, 2:6], selected[prior_index, 2:6]
                )
                if containment >= containment_threshold and area_ratio >= minimum_area_ratio:
                    suppress = True
                    break
        if not suppress:
            kept_indexes.append(index)
    return selected[kept_indexes], [reasons[index] for index in kept_indexes]


def suppress_contained_same_class_duplicates(
    selected: np.ndarray,
    reasons: list[str],
    *,
    containment_threshold: float = SAME_CLASS_CONTAINMENT_THRESHOLD,
    minimum_area_ratio: float = SAME_CLASS_MINIMUM_AREA_RATIO,
) -> tuple[np.ndarray, list[str]]:
    """Remove nested same-class duplicates for every non-Clip detector class.

    Clip retains its historical confidence-first rule and is deliberately
    ignored here. For the other seven classes, boxes are considered from
    largest to smallest, so the outer parent wins even when an inner child has
    higher confidence. By default there is no smaller/larger area-ratio
    safeguard.
    """

    if not len(selected):
        return selected, reasons
    widths = np.maximum(0.0, selected[:, 4] - selected[:, 2])
    heights = np.maximum(0.0, selected[:, 5] - selected[:, 3])
    areas = widths * heights
    # Area is the primary key and confidence is only a deterministic tie-break.
    # np.lexsort uses the last key as the primary key.
    order = np.lexsort((-selected[:, 1], -areas))
    kept_indexes: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        suppress = False
        class_index = int(selected[index, 0])
        if CLASSES[class_index] not in SAME_CLASS_PARENT_ONLY_CLASSES:
            kept_indexes.append(index)
            continue
        for prior_index in kept_indexes:
            if class_index != int(selected[prior_index, 0]):
                continue
            _, containment, area_ratio, _ = overlap_metrics(
                selected[index, 2:6], selected[prior_index, 2:6]
            )
            if containment >= containment_threshold and area_ratio >= minimum_area_ratio:
                suppress = True
                break
        if not suppress:
            kept_indexes.append(index)
    return selected[kept_indexes], [reasons[index] for index in kept_indexes]


def cap_singleton_confirmed_classes(
    selected: np.ndarray,
    reasons: list[str],
    *,
    singleton_classes: frozenset[str] = SINGLETON_CONFIRMED_CLASSES,
) -> tuple[np.ndarray, list[str]]:
    """Keep only the strongest confirmed Specimen and Specimen Bag boxes."""

    if not len(selected):
        return selected, reasons
    singleton_indexes = {CLASS_INDEX[name] for name in singleton_classes}
    order = np.argsort(-selected[:, 1], kind="stable")
    seen_singletons: set[int] = set()
    kept_indexes: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        class_index = int(selected[index, 0])
        if class_index in singleton_indexes:
            if class_index in seen_singletons:
                continue
            seen_singletons.add(class_index)
        kept_indexes.append(index)
    return selected[kept_indexes], [reasons[index] for index in kept_indexes]


def policy_for_class(policy: SelectionPolicy, class_name: str) -> SelectionPolicy:
    """Return the uniform policy with the sole External Drain threshold override."""

    if class_name != "External Drain":
        return policy
    return SelectionPolicy(
        EXTERNAL_DRAIN_THRESHOLD,
        policy.nms_iou,
        policy.candidate_floor,
        policy.minimum_iou,
        policy.maximum_votes,
    )


def select_boxes(all_rows: np.ndarray, policy: SelectionPolicy) -> tuple[np.ndarray, list[str]]:
    """Apply threshold/NMS and one missing-class overlap recovery.

    The recovery is intentionally unavailable once a class has any selected
    object.  Low-confidence consensus can establish presence, but cannot
    inflate an already-positive class count.
    """

    selected = all_rows[all_rows[:, 1] >= policy.threshold]
    if len(selected):
        selected = classwise_nms(selected, policy.nms_iou)
    reasons = ["threshold_nms"] * len(selected)
    recovered: list[np.ndarray] = []
    for class_index in range(len(CLASSES)):
        if np.any(selected[:, 0] == class_index):
            continue
        candidates = all_rows[
            (all_rows[:, 0] == class_index)
            & (all_rows[:, 1] >= policy.candidate_floor)
            & (all_rows[:, 1] < policy.threshold)
        ]
        if len(candidates) < 2:
            continue
        candidates = candidates[np.argsort(-candidates[:, 1])]
        best: np.ndarray | None = None
        for seed in candidates:
            cluster = np.asarray(
                [
                    member
                    for member in candidates
                    if aligned(seed[2:6], member[2:6], policy.minimum_iou)
                ],
                dtype=float,
            ).reshape(-1, 6)
            votes = unique_votes(cluster, policy.maximum_votes)
            if len(votes) < 2:
                continue
            pooled = float(1.0 - np.prod(1.0 - votes[:, 1]))
            if pooled < policy.threshold:
                continue
            weights = votes[:, 1] / votes[:, 1].sum()
            geometry = np.sum(votes[:, 2:6] * weights[:, None], axis=0)
            proposal = np.r_[class_index, pooled, geometry]
            if best is None or proposal[1] > best[1]:
                best = proposal
        if best is not None:
            recovered.append(best)
    if recovered:
        selected = np.vstack([selected, np.asarray(recovered)])
        reasons.extend(["overlap_pool_missing_class"] * len(recovered))

    selected, reasons = suppress_contained_clip_duplicates(selected, reasons)
    selected, reasons = suppress_cross_class_duplicate_queries(selected, reasons)
    selected, reasons = suppress_mutually_exclusive_pair(selected, reasons)

    if len(selected):
        order = np.lexsort(
            (
                selected[:, 5],
                selected[:, 4],
                selected[:, 3],
                selected[:, 2],
                -selected[:, 1],
                selected[:, 0],
            )
        )
        selected = selected[order]
        reasons = [reasons[index] for index in order]
    return selected, reasons


def select_class_boxes(
    all_rows: np.ndarray,
    policy: SelectionPolicy,
) -> tuple[np.ndarray, list[str]]:
    """Select one class and suppress duplicate or excess singleton boxes."""

    selected, reasons = select_boxes(all_rows, policy)
    selected, reasons = suppress_contained_same_class_duplicates(selected, reasons)
    selected, reasons = cap_singleton_confirmed_classes(selected, reasons)
    if len(selected):
        order = np.lexsort(
            (
                selected[:, 5],
                selected[:, 4],
                selected[:, 3],
                selected[:, 2],
                -selected[:, 1],
                selected[:, 0],
            )
        )
        selected = selected[order]
        reasons = [reasons[index] for index in order]
    return selected, reasons


def select_confirmed_boxes_for_candidates(
    all_rows: np.ndarray,
    policy: SelectionPolicy = UNIFORM_POLICY,
) -> tuple[np.ndarray, list[str]]:
    """Select confirmed boxes before the Needle singleton cap."""

    selected_parts: list[np.ndarray] = []
    reasons: list[str] = []
    for class_index, class_name in enumerate(CLASSES):
        rows, row_reasons = select_class_boxes(
            all_rows[all_rows[:, 0] == class_index],
            policy_for_class(policy, class_name),
        )
        if len(rows):
            selected_parts.append(rows)
            reasons.extend(row_reasons)
    selected = np.vstack(selected_parts) if selected_parts else np.empty((0, 6), dtype=float)
    selected, reasons = suppress_cross_class_duplicate_queries(selected, reasons)
    selected, reasons = suppress_mutually_exclusive_pair(selected, reasons)
    if len(selected):
        order = np.lexsort(
            (
                selected[:, 5],
                selected[:, 4],
                selected[:, 3],
                selected[:, 2],
                -selected[:, 1],
                selected[:, 0],
            )
        )
        selected = selected[order]
        reasons = [reasons[index] for index in order]
    return selected, reasons


def select_confirmed_boxes(
    all_rows: np.ndarray,
    policy: SelectionPolicy = UNIFORM_POLICY,
) -> tuple[np.ndarray, list[str]]:
    """Select confirmed boxes and cap Needle at one box."""

    selected, reasons = select_confirmed_boxes_for_candidates(all_rows, policy)
    selected, reasons = cap_singleton_confirmed_classes(
        selected,
        reasons,
        singleton_classes=NEEDLE_SINGLETON_CLASSES,
    )
    if len(selected):
        order = np.lexsort(
            (
                selected[:, 5],
                selected[:, 4],
                selected[:, 3],
                selected[:, 2],
                -selected[:, 1],
                selected[:, 0],
            )
        )
        selected = selected[order]
        reasons = [reasons[index] for index in order]
    return selected, reasons


def select_aggregated_candidates(
    all_rows: np.ndarray,
    confirmed: np.ndarray,
    policy: SelectionPolicy = UNIFORM_POLICY,
    *,
    activation_floor: float = CANDIDATE_ACTIVATION_FLOOR,
    support_floor: float = CANDIDATE_SUPPORT_FLOOR,
) -> list[dict[str, Any]]:
    """Return at most one unconfirmed, aggregated DINO candidate per class.

    A candidate needs one anchor at ``activation_floor``. Once activated, its
    direct spatial cluster inherits same-class support down to
    ``support_floor``. Inherited scores affect only the unweighted mean box;
    displayed confidence remains the strongest anchor confidence.
    """

    output: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASSES):
        threshold = policy_for_class(policy, class_name).threshold
        support_pool = all_rows[
            (all_rows[:, 0] == class_index)
            & (all_rows[:, 1] >= support_floor)
            & (all_rows[:, 1] < threshold)
        ]
        confirmed_class = confirmed[confirmed[:, 0] == class_index]
        support_pool = np.asarray(
            [
                row
                for row in support_pool
                if not any(
                    overlap_metrics(row[2:6], prior[2:6])[0] >= policy.nms_iou
                    for prior in confirmed_class
                )
            ],
            dtype=float,
        ).reshape(-1, 6)
        anchors = support_pool[support_pool[:, 1] >= activation_floor]
        if not len(anchors):
            continue
        clusters: list[np.ndarray] = []
        for seed in anchors[np.argsort(-anchors[:, 1], kind="stable")]:
            cluster = np.asarray(
                [row for row in support_pool if aligned(seed[2:6], row[2:6], policy.minimum_iou)],
                dtype=float,
            ).reshape(-1, 6)
            if len(cluster):
                clusters.append(cluster)
        cluster = max(
            clusters,
            key=lambda rows: (
                float(rows[rows[:, 1] >= activation_floor, 1].max()),
                len(rows),
            ),
        )
        anchor_scores = cluster[cluster[:, 1] >= activation_floor, 1]
        output.append(
            {
                "class_name": class_name,
                "confidence": float(anchor_scores.max()),
                "support": int(len(cluster)),
                "anchor_support": int((cluster[:, 1] >= activation_floor).sum()),
                "inherited_support": int((cluster[:, 1] < activation_floor).sum()),
                "box": np.mean(cluster[:, 2:6], axis=0).tolist(),
                "source": "dinov3_raw",
            }
        )
    return output


def filter_candidate_classes(
    candidates: list[dict[str, Any]],
    confirmed: np.ndarray,
    *,
    fallback_classes: frozenset[str] = FALLBACK_ONLY_CANDIDATE_CLASSES,
    disabled_classes: frozenset[str] = DISABLED_CANDIDATE_CLASSES,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply class-specific candidate gates.

    Gallstone candidates are always removed; above-threshold confirmed
    Gallstone boxes are unaffected. Confirmed boxes of other classes are
    allowed for Specimen, which is retained only when it is the sole candidate
    class and no confirmed Specimen box already exists. For every class, a
    candidate is also removed when at least 90% of its own area lies inside an
    already-confirmed same-class box. This is directed containment: a larger,
    spatially distinct candidate is not removed merely because it contains a
    smaller confirmed box. The returned class names record removals for the
    preprocessing audit.
    """

    candidate_classes = {
        str(item["class_name"])
        for item in candidates
        if str(item["class_name"]) not in disabled_classes
    }
    output: list[dict[str, Any]] = []
    removed: list[str] = []
    for item in candidates:
        class_name = str(item["class_name"])
        if class_name in disabled_classes:
            removed.append(class_name)
            continue
        if class_name in fallback_classes:
            class_index = CLASS_INDEX[class_name]
            same_class_confirmed = bool(np.any(confirmed[:, 0] == class_index))
            if len(candidate_classes) != 1 or same_class_confirmed:
                removed.append(class_name)
                continue
        class_index = CLASS_INDEX[class_name]
        candidate_box = item.get("box")
        confirmed_class = confirmed[confirmed[:, 0] == class_index]
        if candidate_box is not None and any(
            candidate_coverage(np.asarray(candidate_box, dtype=float), prior[2:6])
            >= SAME_CLASS_CANDIDATE_COVERAGE_THRESHOLD
            for prior in confirmed_class
        ):
            removed.append(class_name)
            continue
        output.append(item)
    return output, removed


def filter_cross_class_candidates(
    candidates: list[dict[str, Any]],
    confirmed: np.ndarray,
    *,
    duplicate_iou: float = CANDIDATE_CROSS_CLASS_DUPLICATE_IOU,
    mutually_exclusive_iou: float = MUTUALLY_EXCLUSIVE_IOU,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Remove contradictory cross-class candidate evidence.

    This filter never changes ``confirmed``. A confirmed box always wins over an overlapping
    candidate. Between two candidates, the higher-confidence label wins;
    class order is the deterministic tie-break. The Specimen Bag/Silicone
    Loop rule uses its established 0.80 IoU, while all other cross-class
    conflicts must be near-identical geometry (IoU at least 0.99).
    """

    if not candidates:
        return [], []

    pair_indexes = {CLASS_INDEX[name] for name in MUTUALLY_EXCLUSIVE_PAIR}
    after_confirmed: list[dict[str, Any]] = []
    removals: list[dict[str, str]] = []

    for item in candidates:
        class_name = str(item["class_name"])
        class_index = CLASS_INDEX[class_name]
        candidate_box = np.asarray(item["box"], dtype=float)
        conflict: tuple[str, str] | None = None
        for prior in confirmed:
            prior_index = int(prior[0])
            if prior_index == class_index:
                continue
            iou = overlap_metrics(candidate_box, prior[2:6])[0]
            if class_index in pair_indexes and prior_index in pair_indexes:
                if iou >= mutually_exclusive_iou:
                    conflict = ("bag_loop_confirmed_wins", CLASSES[prior_index])
                    break
            elif iou >= duplicate_iou:
                conflict = ("cross_class_confirmed_wins", CLASSES[prior_index])
                break
        if conflict is None:
            after_confirmed.append(item)
        else:
            reason, conflicting_class = conflict
            removals.append(
                {
                    "reason": reason,
                    "class_name": class_name,
                    "conflicting_class": conflicting_class,
                }
            )

    ordered = sorted(
        after_confirmed,
        key=lambda item: (-float(item["confidence"]), CLASS_INDEX[str(item["class_name"])]),
    )
    kept: list[dict[str, Any]] = []
    for item in ordered:
        class_name = str(item["class_name"])
        class_index = CLASS_INDEX[class_name]
        candidate_box = np.asarray(item["box"], dtype=float)
        conflict = None
        for prior in kept:
            prior_name = str(prior["class_name"])
            prior_index = CLASS_INDEX[prior_name]
            if prior_index == class_index:
                continue
            iou = overlap_metrics(candidate_box, np.asarray(prior["box"], dtype=float))[0]
            if class_index in pair_indexes and prior_index in pair_indexes:
                if iou >= mutually_exclusive_iou:
                    conflict = ("bag_loop_higher_candidate_wins", prior_name)
                    break
            elif iou >= duplicate_iou:
                conflict = ("cross_class_higher_candidate_wins", prior_name)
                break
        if conflict is None:
            kept.append(item)
        else:
            reason, conflicting_class = conflict
            removals.append(
                {
                    "reason": reason,
                    "class_name": class_name,
                    "conflicting_class": conflicting_class,
                }
            )

    kept.sort(key=lambda item: CLASS_INDEX[str(item["class_name"])])
    return kept, removals


def filter_enclosing_candidates(
    candidates: list[dict[str, Any]],
    confirmed: np.ndarray,
    *,
    width: float,
    height: float,
    coverage_threshold: float = SAME_CLASS_CANDIDATE_COVERAGE_THRESHOLD,
    eligible_classes: frozenset[str] = SEMANTIC_ENCLOSING_CANDIDATE_CLASSES,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Remove non-Clip candidates that add no rendered location information.

    The candidate must be larger, cover at least 90% of a same-class confirmed
    box, and render to exactly the same quadrant and two-decimal normalized
    center distance. Clip is excluded deliberately. Confirmed boxes are never
    modified.
    """

    output: list[dict[str, Any]] = []
    removals: list[dict[str, str]] = []
    for item in candidates:
        class_name = str(item["class_name"])
        if class_name not in eligible_classes:
            output.append(item)
            continue
        class_index = CLASS_INDEX[class_name]
        candidate_box = np.asarray(item["box"], dtype=float)
        candidate_area = max(0.0, candidate_box[2] - candidate_box[0]) * max(
            0.0, candidate_box[3] - candidate_box[1]
        )
        candidate_member = np.asarray(
            [class_index, float(item["confidence"]), *candidate_box],
            dtype=float,
        )
        candidate_quadrant, candidate_distance = box_position(
            candidate_member,
            width,
            height,
        )
        conflict = False
        for prior in confirmed[confirmed[:, 0] == class_index]:
            confirmed_area = max(0.0, prior[4] - prior[2]) * max(
                0.0, prior[5] - prior[3]
            )
            if candidate_area <= confirmed_area:
                continue
            confirmed_covered = candidate_coverage(prior[2:6], candidate_box)
            if confirmed_covered < coverage_threshold:
                continue
            confirmed_quadrant, confirmed_distance = box_position(prior, width, height)
            if (
                candidate_quadrant == confirmed_quadrant
                and f"{candidate_distance:.2f}" == f"{confirmed_distance:.2f}"
            ):
                conflict = True
                break
        if conflict:
            removals.append(
                {
                    "reason": "same_class_enclosing_same_rendered_location",
                    "class_name": class_name,
                }
            )
        else:
            output.append(item)
    return output, removals


def candidate_coverage(candidate: np.ndarray, confirmed: np.ndarray) -> float:
    """Return the fraction of candidate area covered by ``confirmed``."""

    x1, y1 = max(candidate[0], confirmed[0]), max(candidate[1], confirmed[1])
    x2, y2 = min(candidate[2], confirmed[2]), min(candidate[3], confirmed[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    candidate_area = max(0.0, candidate[2] - candidate[0]) * max(
        0.0, candidate[3] - candidate[1]
    )
    return intersection / candidate_area if candidate_area else 0.0


def box_position(member: np.ndarray, width: float, height: float) -> tuple[str, float]:
    center_x = (member[2] + member[4]) / 2
    center_y = (member[3] + member[5]) / 2
    quadrant_index = (2 if center_y >= height / 2 else 0) + (1 if center_x >= width / 2 else 0)
    half_diagonal = math.hypot(width, height) / 2
    distance = math.hypot(center_x - width / 2, center_y - height / 2)
    distance = distance / half_diagonal if half_diagonal else 0.0
    return QUADRANTS[quadrant_index], min(1.0, distance)
