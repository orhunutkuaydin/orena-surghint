"""Select video frames and summarize detector tracking evidence."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from . import tracking

TRACKER_CONFIG = {
    "lost_track_buffer": 180,
    "track_activation_threshold": 0.5,
    "high_conf_det_threshold": 0.35,
    "minimum_consecutive_frames": 3,
    "minimum_iou_threshold_first_assoc": 0.02,
    "minimum_iou_threshold_second_assoc": 0.05,
    "minimum_iou_threshold_unconfirmed_assoc": 0.02,
    "buffer_ratio_first": 0.8,
    "buffer_ratio_second": 1.2,
    "instant_first_frame_activation": False,
}

CLASSES = tracking.DETECTOR_CLASSES


@dataclass(frozen=True)
class InputConfig:
    frames: int = 24
    width: int = 768
    height: int = 448


def select_indices(cache, request, cfg, summary=None):
    times = np.asarray(cache["frame_timestamps"], dtype=float)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Empty/nonmonotonic timeline")
    start, end = (float(request["start_time"]), float(request["end_time"]))
    if times[0] < start - 1e-05 or times[-1] >= end + 1e-05:
        raise ValueError("Clip pixels outside request window")
    if len(times) < cfg.frames:
        return np.rint(np.linspace(0, len(times) - 1, cfg.frames)).astype(int).tolist()
    parsed = tracking.parse_segment_question(request["question"])
    summary = summary or tracking.summarize_tracking_window(cache, start, end)
    chosen = []

    def add_index(index):
        if len(chosen) < cfg.frames and int(index) not in chosen:
            chosen.append(int(index))

    def add_time(timestamp):
        add_index(int(np.argmin(np.abs(times - min(times[-1], max(times[0], timestamp))))))

    for index in np.rint(np.linspace(0, len(times) - 1, max(4, cfg.frames // 3))).astype(int):
        add_index(index)
    observed = {row["class_name"] for row in cache["raw_proposals"]}
    use_guidance = parsed.detector_support != "none" and (
        not parsed.target_classes or bool(observed.intersection(parsed.target_classes))
    )
    guided = tracking.select_evidence_frames(summary, request["question"], budget=cfg.frames) if use_guidance else ()
    extra = []
    for target in parsed.target_classes:
        raw_times = [row["time_seconds"] for row in cache["raw_proposals"] if row["class_name"] == target]
        if not raw_times:
            continue
        boundary = (
            min(raw_times) if parsed.operation in ("first_visible", "quadrant_first_visible")
            else max(raw_times) if parsed.operation in ("last_visible", "quadrant_last_visible")
            else None
        )
        if boundary is not None:
            extra += [(boundary + offset, 109 if offset == 0 else 102) for offset in (-0.2, 0, 0.2)]
    candidates = [(frame.timestamp, frame.score) for frame in guided if frame.reason != "uniform coverage"] + extra
    candidates += [(time + offset, 110 if offset == 0 else 105) for time in parsed.timestamps for offset in (-0.2, 0, 0.2)]
    for timestamp, score in sorted(candidates, key=lambda item: (-item[1], item[0])):
        add_time(timestamp)
    while len(chosen) < cfg.frames:
        distance = np.min(np.abs(times[:, None] - times[np.asarray(chosen)][None, :]), axis=1)
        add_index(int(np.argmax(distance)))
    return sorted(chosen)


def focused_hint(cache, summary, request, indices):
    """Describe observed detections and track quality without assigning an answer."""
    parsed = tracking.parse_segment_question(request["question"])
    times = [cache["frame_timestamps"][index] for index in indices]
    classmap = summary.class_map()
    targets = list(parsed.target_classes)
    if not targets:
        targets = [item.class_name for item in sorted(summary.classes, key=lambda item: item.quality_score, reverse=True)[:3]]
    targets = targets[:3]

    def clock(timestamp):
        return tracking.seconds_to_timestamp_precise(float(timestamp))

    def anchor(timestamp):
        index = min(range(len(times)), key=lambda index: abs(times[index] - timestamp))
        return f"Frame {index + 1:02d} at {clock(times[index])}" if abs(times[index] - timestamp) <= 0.31 else "between displayed frames"

    lines = [
        "DETECTOR TRACKING EVIDENCE (RF-DETR predictions, not ground truth). All times are ABSOLUTE procedure time.",
        "Every 5-fps frame was analyzed. Retained tracks passed temporal-quality checks, but may still be wrong. Identities can fragment or switch; track count is not the number of insertions. Appearance/disappearance is not proof of insertion/retrieval. Missing detections are unknown, not absence.",
        f"Question operation={parsed.operation}; clip={clock(summary.start)}--{clock(summary.end)}; detector threshold=0.50.",
        "Detected classes in this clip: " + (", ".join(sorted({row["class_name"] for row in cache["raw_proposals"]})) or "none") + ".",
    ]
    for target in targets:
        if target not in CLASSES:
            lines.append(f"{target}: detector UNSUPPORTED; use visual evidence.")
            continue
        raw = [row for row in cache["raw_proposals"] if row["class_name"] == target]
        if not raw:
            lines.append(f"{target}: no confident detections; visual verification required.")
            continue
        by_frame = {}
        for row in raw:
            by_frame[row["sample_index"]] = by_frame.get(row["sample_index"], 0) + 1
        first, last = min(row["time_seconds"] for row in raw), max(row["time_seconds"] for row in raw)
        dense_duration = sum(
            max(0.0, min(cache["frame_timestamps"][index] + 0.2, summary.end) - cache["frame_timestamps"][index])
            for index in by_frame
        )
        lines.append(
            f"{target}: raw first={clock(first)} ({anchor(first)}); raw last={clock(last)} ({anchor(last)}); detected duration={dense_duration:.1f}s; detected fraction={100 * dense_duration / (summary.end - summary.start):.1f}%; maximum observed simultaneous boxes={max(by_frame.values())} (may include false positives)."
        )
        item = classmap.get(target)
        retained = [] if item is None else [track for track in item.tracks if track.quality.retained]
        lines.append(f"{target}: retained tracklets={len(retained)}; rejected/fragmented candidates remain uncertain.")
        if item is not None and retained:
            intervals = list(item.visible_intervals)
            if parsed.operation in ("last_visible", "quadrant_last_visible", "retrieval_after_time"):
                intervals = intervals[-6:]
            else:
                intervals = intervals[:6]
            spans = "; ".join(f"{clock(interval.start)}--{clock(interval.end)}" for interval in intervals)
            lines.append(
                f"{target}: track-supported presence spans (up to 6 of {len(item.visible_intervals)}; gaps <=1s bridged): {spans}. Bridged presence duration={item.visible_duration:.1f}s; retained maximum simultaneous={item.max_simultaneous}."
            )
            lines.append(
                f"{target}: first-track position={item.first_position}; last-track position={item.last_position}; positions use the ORIGINAL image center, excluding the time header."
            )
        if parsed.timestamps:
            for query in parsed.timestamps[:2]:
                nearest = min(range(len(cache["frame_timestamps"])), key=lambda index: abs(cache["frame_timestamps"][index] - query))
                if abs(cache["frame_timestamps"][nearest] - query) <= 0.31:
                    matches = [row for row in raw if row["sample_index"] == nearest]
                    positions = [tracking._position(row["xyxy"], cache["width"], cache["height"]) for row in matches]
                    lines.append(
                        f"{target}: near requested {clock(query)}: confident boxes={len(matches)}; positions={', '.join(positions) or 'unknown'} ({anchor(cache['frame_timestamps'][nearest])})."
                    )
        if parsed.operation in ("last_visible", "quadrant_last_visible", "retrieval_after_time"):
            retained = sorted(retained, key=lambda track: track.last_seen)[-4:]
        for track in sorted(retained, key=lambda track: track.first_seen)[:4]:
            lines.append(
                f"{target} track {track.track_id}: stable-first estimate={clock(track.first_seen)} ({anchor(track.first_seen)}); last matched={clock(track.last_seen)} ({anchor(track.last_seen)}); matched frames={track.matched_frames}; quality={track.quality.score:.2f}; continuity={track.quality.continuity:.2f}."
            )
    if len(parsed.target_classes) == 2 and all(target in CLASSES for target in parsed.target_classes):
        sets = [{row["sample_index"] for row in cache["raw_proposals"] if row["class_name"] == target} for target in parsed.target_classes]
        common = sorted(sets[0] & sets[1])
        duration = sum(min(0.2, summary.end - cache["frame_timestamps"][index]) for index in common)
        lines.append(
            f"Both named targets detected in the same frames: duration={duration:.1f}s; "
            + (f"first={clock(cache['frame_timestamps'][common[0]])}; last={clock(cache['frame_timestamps'][common[-1]])}." if common else "none confidently detected together.")
        )
    return "\n".join(lines)


def question_evidence(cache, request, cfg):
    summary = tracking.summarize_tracking_window(cache, float(request["start_time"]), float(request["end_time"]))
    indices = select_indices(cache, request, cfg, summary)
    return {"indices": indices, "times": [cache["frame_timestamps"][index] for index in indices],
            "hint": focused_hint(cache, summary, request, indices)}


CONFIG = InputConfig()
