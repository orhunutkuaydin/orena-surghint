"""Decode question clips and associate detections on their original timeline."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from . import hints as core
from .rendering import EvidenceConfig, render_video


def open_video(path):
    import torch  # Import before Decord to initialize the shared runtime.
    import decord

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=2)
    validate_fps(reader)
    return reader


def track_video(detector, reader, request):
    from .detector import BATCH_SIZE

    relative, indices = window_indices(reader, request)
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Invalid dense clip timeline")
    detections = []
    for offset in range(0, len(indices), BATCH_SIZE):
        frames = reader.get_batch(
            indices[offset : offset + BATCH_SIZE].tolist()
        ).asnumpy()
        detections.extend(detector.predict(frames))
    height, width = frames.shape[1:3]
    return associate(
        relative[indices] + float(request["start_time"]),
        detections,
        width,
        height,
        request["videoID"],
    )


def visuals(reader, tracking, request):
    relative, indices = window_indices(reader, request)
    if len(indices) != tracking["analyzed_frames"]:
        raise ValueError("Clip changed relative to tracking evidence")
    absolute = relative + float(request["start_time"])
    tracked_times = np.asarray(tracking["frame_timestamps"], dtype=float)
    if (
        tracked_times.shape != indices.shape
        or not np.isfinite(tracked_times).all()
        or not np.allclose(absolute[indices], tracked_times, atol=1e-6, rtol=0)
    ):
        raise ValueError("Decoded timestamps differ from tracking evidence")
    evidence = core.question_evidence(tracking, request, core.CONFIG)
    raw = reader.get_batch(evidence["indices"]).asnumpy()
    if tuple(raw.shape[1:3]) != (tracking["height"], tracking["width"]):
        raise ValueError("Decoded geometry differs from detector boxes")
    if not np.allclose(
        absolute[evidence["indices"]], evidence["times"], atol=1e-6, rtol=0
    ):
        raise ValueError("Selected decoded timestamps differ from tracking evidence")
    by_frame = {}
    for proposal in tracking["raw_proposals"]:
        by_frame.setdefault(proposal["sample_index"], []).append(proposal)
    scale = np.array([tracking["width"], tracking["height"]] * 2)
    detections = []
    for index in evidence["indices"]:
        proposals = by_frame.get(index, [])
        detections.append(
            dict(
                boxes=[(np.asarray(p["xyxy"]) / scale).tolist() for p in proposals],
                classes=[p["class_id"] for p in proposals],
                scores=[p["confidence"] for p in proposals],
            )
        )
    frames = render_video(raw, detections, evidence["times"], EvidenceConfig())
    evidence["total_frames"] = len(reader)
    return evidence, frames


def validate_fps(reader):
    fps = float(reader.get_avg_fps())
    if len(reader) < 1 or not np.isfinite(fps) or (not 4.8 <= fps <= 5.2):
        raise ValueError(
            f"Expected nonempty approximately 5-fps platform clip; FPS={fps}"
        )
    return fps


def video_times(reader):
    validate_fps(reader)
    raw = np.asarray(
        reader.get_frame_timestamp(list(range(len(reader)))), dtype=np.float64
    )
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite decoded frame timestamp table")
    if raw.ndim == 2 and raw.shape[1] == 2:
        raw = raw[:, 0]
    if raw.ndim != 1 or len(raw) != len(reader) or (not np.isfinite(raw).all()):
        raise ValueError("Invalid decoded frame timestamp table")
    relative = raw - raw[0]
    if not np.isfinite(relative).all() or np.any(np.diff(relative) <= 0):
        raise ValueError("Nonfinite/nonmonotonic decoded frame timestamps")
    return relative


def window_indices(reader, request):
    relative = video_times(reader)
    start, end = (float(request["start_time"]), float(request["end_time"]))
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError("Invalid request window")
    duration = end - start
    period = 1.0 / validate_fps(reader)
    tolerance = period + 0.01
    if (
        relative[-1] > duration + tolerance
        or relative[-1] + period < duration - tolerance
    ):
        raise ValueError("Decoded clip duration does not match the request window")
    indices = np.flatnonzero(relative < duration - 1e-06)
    if not len(indices):
        raise ValueError("No decoded frames inside request window")
    return (relative, indices)


def associate(times, frame_detections, width, height, video_id=""):
    """Associate detections by class, updating C-BIoU on every frame."""
    from trackers import CBIoUTracker
    import supervision as sv

    times = np.asarray(times, dtype=float)
    if len(times) != len(frame_detections) or len(times) < 1:
        raise ValueError("Detection/time mismatch")
    if not np.isfinite(times).all() or np.any(times < 0) or width <= 0 or (height <= 0):
        raise ValueError("Invalid timestamps/geometry")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Nonmonotonic decoded timestamps")
    trackers = {
        i: CBIoUTracker(frame_rate=5, **core.TRACKER_CONFIG)
        for i in range(len(core.CLASSES))
    }
    observations, raw = ([], [])
    for i, (timestamp, row) in enumerate(zip(times, frame_detections, strict=True)):
        boxes = np.asarray(row["boxes"], dtype=np.float32).reshape(-1, 4)
        confidence = np.asarray(row["scores"], dtype=np.float32)
        classes = np.asarray(row["classes"], dtype=int)
        if len(boxes) != len(confidence) or len(boxes) != len(classes):
            raise ValueError("Malformed detections")
        if (
            not np.isfinite(boxes).all()
            or not np.isfinite(confidence).all()
            or np.any(boxes < 0)
            or np.any(boxes > 1)
            or np.any(confidence < 0.5 - 1e-06)
            or np.any(confidence > 1)
            or np.any(boxes[:, 2:] <= boxes[:, :2])
            or np.any(classes < 0)
            or np.any(classes >= len(core.CLASSES))
        ):
            raise ValueError("Unexpected normalized detection values/class order")
        boxes = boxes * np.array([width, height, width, height], dtype=np.float32)
        detections = sv.Detections(xyxy=boxes, confidence=confidence, class_id=classes)
        for b, score, c in zip(boxes, confidence, classes, strict=True):
            raw.append(
                {
                    "sample_index": i,
                    "source_frame": i,
                    "time_seconds": float(timestamp),
                    "class_id": int(c),
                    "class_name": core.CLASSES[c],
                    "confidence": float(score),
                    "xyxy": b.astype(float).tolist(),
                }
            )
        for c, tracker in trackers.items():
            tracked = tracker.update(detections[detections.class_id == c])
            if tracked.tracker_id is None:
                continue
            for b, score, identity in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id, strict=True
            ):
                if int(identity) >= 0:
                    observations.append(
                        {
                            "sample_index": i,
                            "source_frame": i,
                            "time_seconds": float(timestamp),
                            "class_id": c,
                            "class_name": core.CLASSES[c],
                            "track_id": int(identity),
                            "confidence": float(score),
                            "xyxy": b.astype(float).tolist(),
                        }
                    )
    return {
        "schema_version": 1,
        "video_id": str(video_id),
        "width": int(width),
        "height": int(height),
        "analysis_fps": 5.0,
        "class_names": list(core.CLASSES),
        "tracker": {**core.TRACKER_CONFIG, "frame_rate": 5},
        "input_frames": len(times),
        "analyzed_frames": len(times),
        "frame_timestamps": times.tolist(),
        "observations": observations,
        "raw_proposals": raw,
    }
