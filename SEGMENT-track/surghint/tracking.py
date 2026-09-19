"""Temporal track summaries and question-aware frame selection."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any

DETECTOR_CLASSES = (
    "Sponge",
    "Clip",
    "Specimen Bag",
    "Silicone Loop",
    "External Drain",
    "Needle",
    "Gallstone",
    "Specimen",
)
_ALIASES = {
    "Sponge": ("sponge", "sponges", "swab", "swabs", "gauze"),
    "Clip": ("clip", "clips"),
    "Specimen Bag": (
        "specimen bag",
        "specimen bags",
        "retrieval bag",
        "retrieval bags",
        "extraction bag",
        "extraction bags",
    ),
    "Silicone Loop": ("silicone loop", "silicone loops", "vessel loop", "vessel loops"),
    "External Drain": ("external drain", "external drains", "drain", "drains"),
    "Needle": ("needle", "needles"),
    "Gallstone": ("gallstone", "gallstones", "stone", "stones"),
    "Specimen": ("specimen", "specimens"),
    "Mesh": ("mesh",),
    "Absorbable Hemostatic Agent": (
        "absorbable hemostatic agent",
        "haemostatic agent",
        "hemostatic agent",
    ),
}


@dataclass(frozen=True)
class PresenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TrackSummary:
    class_name: str
    track_id: int
    first_seen: float
    last_seen: float
    matched_frames: int
    first_position: str
    last_position: str
    best_times: tuple[float, ...]
    quality: TrackQuality


@dataclass(frozen=True)
class TrackQuality:
    """Auditable, class-agnostic temporal evidence for one object track."""

    continuity: float
    score: float
    retained: bool


@dataclass(frozen=True)
class ClassSummary:
    class_name: str
    tracks: tuple[TrackSummary, ...]
    visible_intervals: tuple[PresenceInterval, ...]
    first_seen: float
    last_seen: float
    visible_duration: float
    max_simultaneous: int
    max_count_times: tuple[float, ...]
    quality_score: float
    retained_tracks: int
    first_position: str
    last_position: str


@dataclass(frozen=True)
class WindowTrackingSummary:
    start: float
    end: float
    analysis_fps: float
    classes: tuple[ClassSummary, ...]

    def class_map(self) -> dict[str, ClassSummary]:
        return {item.class_name: item for item in self.classes}


@dataclass(frozen=True)
class EvidenceFrame:
    timestamp: float
    reason: str
    score: float


@dataclass(frozen=True)
class ParsedSegmentQuestion:
    """Online-safe interpretation derived from question text alone."""

    operation: str
    target_classes: tuple[str, ...]
    timestamps: tuple[float, ...]
    detector_support: str


def _position(box: list[float], width: float, height: float) -> str:
    x1, y1, x2, y2 = box
    horizontal = "left" if (x1 + x2) / 2 < width / 2 else "right"
    vertical = "top" if (y1 + y2) / 2 < height / 2 else "bottom"
    return f"{vertical}/{horizontal}"


def _box_geometry(box: list[float]) -> tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(1e-06, x2 - x1)
    height = max(1e-06, y2 - y1)
    return ((x1 + x2) / 2, (y1 + y2) / 2, width * height, width / height, width, height)


def _box_iou(one: list[float], two: list[float]) -> float:
    x1 = max(float(one[0]), float(two[0]))
    y1 = max(float(one[1]), float(two[1]))
    x2 = min(float(one[2]), float(two[2]))
    y2 = min(float(one[3]), float(two[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    one_area = max(0.0, float(one[2]) - float(one[0])) * max(
        0.0, float(one[3]) - float(one[1])
    )
    two_area = max(0.0, float(two[2]) - float(two[0])) * max(
        0.0, float(two[3]) - float(two[1])
    )
    union = one_area + two_area - intersection
    return intersection / union if union > 0 else 0.0


def _symmetric_ratio(one: float, two: float) -> float:
    smaller = max(1e-06, min(one, two))
    return max(one, two) / smaller


def split_discontinuous_tracks(
    observations: list[dict[str, Any]],
    analysis_fps: float,
    *,
    max_observation_gap_seconds: float = 1.0,
    maximum_iou: float = 0.35,
    maximum_normalized_center_shift: float = 0.25,
    maximum_area_ratio: float = 3.0,
    maximum_aspect_ratio: float = 2.0,
    minimum_confidence_delta: float = 0.45,
    minimum_discontinuity_cues: int = 3,
) -> list[dict[str, Any]]:
    if analysis_fps <= 0 or max_observation_gap_seconds <= 0:
        raise ValueError("FPS and maximum observation gap must be positive")
    if not 0 <= maximum_iou <= 1 or not 0 <= minimum_confidence_delta <= 1:
        raise ValueError("IoU and confidence thresholds must lie in [0, 1]")
    if (
        maximum_normalized_center_shift <= 0
        or maximum_area_ratio <= 1
        or maximum_aspect_ratio <= 1
        or (minimum_discontinuity_cues < 1)
    ):
        raise ValueError("Geometry thresholds and cue count must be positive")
    if not observations:
        return []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    next_track_id: dict[str, int] = {}
    for row in observations:
        class_name = str(row["class_name"])
        track_id = int(row["track_id"])
        grouped.setdefault((class_name, track_id), []).append(row)
        next_track_id[class_name] = max(next_track_id.get(class_name, 0), track_id + 1)
    adjusted: list[dict[str, Any]] = []
    for (class_name, source_track_id), source_rows in sorted(grouped.items()):
        source_rows.sort(key=lambda row: float(row["time_seconds"]))
        assigned_track_id = source_track_id
        segment = 0
        previous: dict[str, Any] | None = None
        for row in source_rows:
            if previous is not None:
                gap_seconds = float(row["time_seconds"]) - float(
                    previous["time_seconds"]
                )
                previous_geometry = _box_geometry(previous["xyxy"])
                current_geometry = _box_geometry(row["xyxy"])
                previous_diagonal = max(
                    1e-06, math.hypot(previous_geometry[4], previous_geometry[5])
                )
                iou = _box_iou(previous["xyxy"], row["xyxy"])
                center_shift = (
                    math.dist(previous_geometry[:2], current_geometry[:2])
                    / previous_diagonal
                )
                area_ratio = _symmetric_ratio(previous_geometry[2], current_geometry[2])
                aspect_ratio = _symmetric_ratio(
                    previous_geometry[3], current_geometry[3]
                )
                confidence_delta = abs(
                    float(row["confidence"]) - float(previous["confidence"])
                )
                cues: list[str] = []
                if iou < maximum_iou:
                    cues.append("low_iou")
                if center_shift > maximum_normalized_center_shift:
                    cues.append("center_jump")
                if area_ratio > maximum_area_ratio:
                    cues.append("area_jump")
                if aspect_ratio > maximum_aspect_ratio:
                    cues.append("aspect_jump")
                if confidence_delta > minimum_confidence_delta:
                    cues.append("confidence_jump")
                if (
                    gap_seconds > max_observation_gap_seconds
                    and len(cues) >= minimum_discontinuity_cues
                ):
                    new_track_id = next_track_id[class_name]
                    next_track_id[class_name] += 1
                    assigned_track_id = new_track_id
                    segment += 1
            copied = dict(row)
            copied["source_track_id"] = source_track_id
            copied["track_segment"] = segment
            copied["track_id"] = assigned_track_id
            adjusted.append(copied)
            previous = row
    adjusted.sort(
        key=lambda row: (
            float(row["time_seconds"]),
            int(row["class_id"]),
            int(row["track_id"]),
        )
    )
    return adjusted


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def score_track_observations(
    rows: list[dict[str, Any]],
    analysis_fps: float,
    *,
    high_confidence_threshold: float = 0.35,
    temporal_window_seconds: float = 1.0,
) -> TrackQuality:
    if not rows:
        raise ValueError("Cannot score an empty track")
    if analysis_fps <= 0 or temporal_window_seconds <= 0:
        raise ValueError("FPS and temporal window must be positive")
    if not 0 <= high_confidence_threshold <= 1:
        raise ValueError("High-confidence threshold must lie in [0, 1]")
    confidence_by_frame: dict[int, float] = {}
    for row in rows:
        frame = round(float(row["time_seconds"]) * analysis_fps)
        confidence = min(1.0, max(0.0, float(row["confidence"])))
        confidence_by_frame[frame] = max(
            confidence_by_frame.get(frame, 0.0), confidence
        )
    first_frame, last_frame = (min(confidence_by_frame), max(confidence_by_frame))
    dense = [
        confidence_by_frame.get(frame, 0.0)
        for frame in range(first_frame, last_frame + 1)
    ]
    matched = sorted(confidence_by_frame)
    matched_confidences = [confidence_by_frame[frame] for frame in matched]
    span_frames = len(dense)
    continuity = len(matched) / span_frames
    longest_run = 1
    current_run = 1
    for previous, current in zip(matched, matched[1:]):
        if current == previous + 1:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
    longest_run_fraction = longest_run / len(matched)
    window_frames = max(1, round(temporal_window_seconds * analysis_fps))
    if len(dense) < window_frames:
        rolling_sums = [sum(dense)]
    else:
        prefix = [0.0]
        for value in dense:
            prefix.append(prefix[-1] + value)
        rolling_sums = [
            prefix[index + window_frames] - prefix[index]
            for index in range(len(dense) - window_frames + 1)
        ]
    sustained_confidence = max(rolling_sums) / window_frames
    lower_quartile = _quantile(matched_confidences, 0.25)
    confidence_evidence = math.sqrt(lower_quartile * sustained_confidence)
    interruption_penalty = continuity * math.sqrt(longest_run_fraction)
    score = confidence_evidence * interruption_penalty
    strict_retention = (
        lower_quartile >= high_confidence_threshold
        and sustained_confidence >= high_confidence_threshold
        and (score >= high_confidence_threshold)
    )
    long_sustained_retention = (
        span_frames >= 5 * window_frames
        and lower_quartile >= 0.9 * high_confidence_threshold
        and (sustained_confidence >= min(1.0, 2.0 * high_confidence_threshold))
        and (continuity >= 0.8)
        and (longest_run_fraction >= 0.5)
        and (score >= high_confidence_threshold)
    )
    retained = strict_retention or long_sustained_retention
    return TrackQuality(continuity=continuity, score=score, retained=retained)


def _merge_presence(
    times: list[float], start: float, end: float, frame_period: float, max_gap: float
) -> tuple[PresenceInterval, ...]:
    if not times:
        return ()
    groups: list[list[float]] = [[times[0]]]
    for value in times[1:]:
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(
        (
            PresenceInterval(
                max(start, group[0] - frame_period), min(end, group[-1] + frame_period)
            )
            for group in groups
        )
    )


def summarize_tracking_window(
    cache: dict[str, Any], start: float, end: float, *, max_gap_seconds: float = 1.0
) -> WindowTrackingSummary:
    """Summarize retained tracks and their visibility within the clip."""
    if not 0 <= start < end:
        raise ValueError("Tracking window must satisfy 0 <= start < end")
    analysis_fps = float(cache["analysis_fps"])
    if analysis_fps <= 0:
        raise ValueError("Tracking cache has invalid analysis_fps")
    frame_period = 1.0 / analysis_fps
    width, height = (float(cache["width"]), float(cache["height"]))
    class_names = tuple(cache.get("class_names") or DETECTOR_CLASSES)
    observations = [
        row
        for row in cache["observations"]
        if start <= float(row["time_seconds"]) <= end
        and str(row["class_name"]) in class_names
    ]
    observations = split_discontinuous_tracks(observations, analysis_fps)
    observations.sort(
        key=lambda row: (
            str(row["class_name"]),
            int(row["track_id"]),
            float(row["time_seconds"]),
        )
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in observations:
        key = (str(row["class_name"]), int(row["track_id"]))
        grouped.setdefault(key, []).append(row)
    minimum_consecutive = int(
        cache.get("tracker", {}).get("minimum_consecutive_frames", 3)
    )
    high_confidence_threshold = float(
        cache.get("tracker", {}).get("high_conf_det_threshold", 0.35)
    )
    backdate = max(0, minimum_consecutive - 1) * frame_period
    track_summaries: list[TrackSummary] = []
    for (class_name, track_id), rows in grouped.items():
        times = [float(row["time_seconds"]) for row in rows]
        ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
        track_segment = int(rows[0].get("track_segment", 0))
        first_seen = times[0] if track_segment else max(start, times[0] - backdate)
        track_summaries.append(
            TrackSummary(
                class_name=class_name,
                track_id=track_id,
                first_seen=first_seen,
                last_seen=times[-1],
                matched_frames=len(rows),
                first_position=_position(rows[0]["xyxy"], width, height),
                last_position=_position(rows[-1]["xyxy"], width, height),
                best_times=tuple((float(row["time_seconds"]) for row in ranked[:3])),
                quality=score_track_observations(
                    rows,
                    analysis_fps,
                    high_confidence_threshold=high_confidence_threshold,
                ),
            )
        )
    classes: list[ClassSummary] = []
    for class_name in class_names:
        tracks = sorted(
            (track for track in track_summaries if track.class_name == class_name),
            key=lambda track: (track.first_seen, track.track_id),
        )
        if not tracks:
            continue
        retained = [track for track in tracks if track.quality.retained]
        evidence_tracks = retained or tracks
        evidence_keys = {
            (track.class_name, track.track_id) for track in evidence_tracks
        }
        class_rows = [
            row
            for row in observations
            if (str(row["class_name"]), int(row["track_id"])) in evidence_keys
        ]
        times = sorted({float(row["time_seconds"]) for row in class_rows})
        intervals = _merge_presence(
            times, start, end, frame_period, max(max_gap_seconds, 2 * frame_period)
        )
        simultaneous: dict[float, set[int]] = {}
        for row in class_rows:
            simultaneous.setdefault(float(row["time_seconds"]), set()).add(
                int(row["track_id"])
            )
        max_simultaneous = max((len(ids) for ids in simultaneous.values()), default=0)
        max_count_times = tuple(
            (time for time, ids in simultaneous.items() if len(ids) == max_simultaneous)
        )[:5]
        first_track = min(evidence_tracks, key=lambda track: track.first_seen)
        last_track = max(evidence_tracks, key=lambda track: track.last_seen)
        visible_duration = sum((interval.duration for interval in intervals))
        classes.append(
            ClassSummary(
                class_name=class_name,
                tracks=tuple(tracks),
                visible_intervals=intervals,
                first_seen=min((track.first_seen for track in evidence_tracks)),
                last_seen=max((track.last_seen for track in evidence_tracks)),
                visible_duration=visible_duration,
                max_simultaneous=max_simultaneous,
                max_count_times=max_count_times,
                quality_score=max((track.quality.score for track in evidence_tracks)),
                retained_tracks=len(retained),
                first_position=first_track.first_position,
                last_position=last_track.last_position,
            )
        )
    return WindowTrackingSummary(
        start=start, end=end, analysis_fps=analysis_fps, classes=tuple(classes)
    )


def classes_in_question(question: str) -> tuple[str, ...]:
    text = question.casefold()
    text = re.sub("\\(\\s*in case of a specimen\\s*\\)", "", text)
    if (
        re.search("\\b(?:which|what)\\b.*\\bforeign object\\b", text)
        and "class name" in text
        and ("none" in text)
    ):
        return ()
    candidates: list[tuple[int, int, str]] = []
    for class_name, aliases in _ALIASES.items():
        for alias in aliases:
            for match in re.finditer(f"\\b{re.escape(alias)}\\b", text):
                candidates.append((match.start(), match.end(), class_name))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    occupied: list[tuple[int, int]] = []
    found: list[str] = []
    for start, end, class_name in candidates:
        if any(
            (start < used_end and used_start < end for used_start, used_end in occupied)
        ):
            continue
        occupied.append((start, end))
        if class_name not in found:
            found.append(class_name)
    return tuple(found)


def question_timestamps(question: str) -> tuple[float, ...]:
    """Return every valid absolute ``hh:mm:ss`` timestamp in textual order."""
    result: list[float] = []
    for match in re.finditer("\\b(\\d{1,2}):(\\d{2}):(\\d{2})\\b", question):
        hours, minutes, seconds = (int(value) for value in match.groups())
        if minutes >= 60 or seconds >= 60:
            continue
        value = float(hours * 3600 + minutes * 60 + seconds)
        if value not in result:
            result.append(value)
    return tuple(result)


def _question_operation(question: str) -> str:
    text = re.sub("\\s+", " ", question.strip()).casefold()
    timestamps = question_timestamps(question)
    if (
        "how much time passes between when the first" in text
        and "second, distinct" in text
    ):
        return "second_instance_delay"
    if "last visible just before" in text and "re-appear later" in text:
        return "reappears_after_time"
    if (
        text.startswith("after the first ")
        and "which other foreign object classes are visible" in text
    ):
        return "classes_visible_after_action"
    if text.startswith("after the ") and "were inserted or created" in text:
        return "classes_action_after_action"
    if text.startswith("before the ") and "was there already" in text:
        return "other_class_visible_before_action"
    if "how many" in text and "are inserted in the abdomen" in text:
        return "action_count"
    if "how many" in text and "are retrieved from the surgical site" in text:
        return "retrieval_count"
    if text.startswith("added up, what was the total count"):
        return "named_class_sum"
    if "class, which is last seen" in text and "co-occur" in text:
        return "classes_cooccurring_with_last"
    if "visible for the longest" in text and "total duration" in text:
        return "longest_duration_class"
    if "at which time points were" in text and "applied" in text:
        return "all_action_times"
    if (
        text.startswith("during the video, at what time is the")
        and "inserted in the abdomen for the first time" in text
    ):
        return "ordinal_action_time"
    if "when is it retrieved" in text or "when was it retrieved" in text:
        return "retrieval_after_time"
    if "first inserted" in text or "inserted in the abdomen for the first time" in text:
        return "first_action_time"
    if "first created" in text:
        return "first_action_time"
    if "being inserted" in text and timestamps:
        return "action_at_time"
    if "being retrieved" in text and timestamps:
        return "retrieval_at_time"
    if "inserted or created" in text and timestamps:
        return "action_class_at_time"
    if "inserted or created" in text:
        return "single_action_class"
    if "last visible at the same time" in text:
        return "last_co_visible"
    if "first visible at the same time" in text:
        return "first_co_visible"
    if "relative to the image center" in text and "last time" in text:
        return "quadrant_last_visible"
    if "in which quadrant" in text and (
        "first appears" in text or "first becomes visible" in text
    ):
        return "quadrant_first_visible"
    if re.search("\\blast\\b.*\\bvisible\\b|\\blast appearance\\b", text):
        return "last_visible"
    if re.search(
        "\\bfirst\\b.*\\bvisible\\b|\\bfirst appearance\\b|\\bfirst appears\\b", text
    ):
        return "first_visible"
    if "leave the surgical view for at least three seconds the first time" in text:
        return "first_disappearance"
    if "leave the field of view for at least" in text and "re-enter later" in text:
        return "reentry_exists"
    if "separate times" in text and "return" in text:
        return "reentry_count"
    if "also appear at" in text and len(timestamps) >= 2:
        return "presence_at_two_times"
    if "co-occur in any frame" in text:
        return "cooccurs"
    if (
        "only foreign object class" in text
        or "only foreign object class on screen" in text
    ):
        return "target_only_duration"
    if "visible at the same time" in text and "adding all time intervals" in text:
        return "co_visible_duration"
    if "for how long" in text or "how long" in text or "total duration" in text:
        return "visible_duration"
    if "how many of the frames" in text and "contain" in text:
        return "frame_percentage"
    if "maximum number" in text and "appearing at once" in text:
        return "max_simultaneous"
    if text.startswith("in total, how many distinct"):
        return "validated_instances"
    if "different foreign object classes" in text:
        return "candidate_class_count"
    if "distinct foreign object instances" in text:
        return (
            "instances_excluding_class"
            if "do not count" in text
            else "validated_instances"
        )
    if "types of foreign objects are seen between" in text:
        return "class_set_interval"
    if "which foreign object classes appear" in text:
        return "class_set"
    if "what surgical foreign object is visible" in text:
        return "class_set"
    if "unique class of foreign object appearing" in text:
        return "ordinal_visible_class"
    if "have not been populated" in text:
        return "unpopulated_quadrants"
    if "have been populated" in text:
        return "populated_quadrants"
    if "for most of the time" in text or "for the most time" in text:
        return "quadrant_most_dwell"
    if "longest continuous time" in text:
        return "quadrant_longest_run"
    if "at time point" in text or "at timepoint" in text:
        return (
            "objects_at_time"
            if "all relative central positions" in text
            else "quadrant_at_time"
        )
    if timestamps and re.search("\\bvisible\\b", text):
        return "presence_at_time"
    if "color" in text or "colour" in text:
        return "visual_attribute"
    if "organ" in text or "anatomical" in text:
        return "anatomy"
    return "generic"


def parse_segment_question(question: str) -> ParsedSegmentQuestion:
    """Identify question targets and temporal cues for visual evidence selection."""
    targets = classes_in_question(question)
    support = (
        "none"
        if any((name not in DETECTOR_CLASSES for name in targets))
        else "full"
        if targets
        else "partial"
    )
    return ParsedSegmentQuestion(
        _question_operation(question), targets, question_timestamps(question), support
    )


def high_confidence_classes(summary: WindowTrackingSummary) -> tuple[ClassSummary, ...]:
    """Return classes with at least one track passing the temporal gate."""
    return tuple((item for item in summary.classes if item.retained_tracks > 0))


def seconds_to_timestamp_precise(seconds: float) -> str:
    """Render absolute seconds at the 5-fps precision visible to the model."""
    tenths = max(0, int(round(float(seconds) * 10)))
    whole, decimal = divmod(tenths, 10)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{decimal}"


def select_evidence_frames(
    summary: WindowTrackingSummary, question: str, *, budget: int = 8
) -> tuple[EvidenceFrame, ...]:
    """Select temporal anchors plus question-specific track evidence."""
    if budget < 1:
        raise ValueError("Evidence-frame budget must be positive")
    parsed = parse_segment_question(question)
    question_type = parsed.operation
    requested = parsed.target_classes
    class_map = summary.class_map()
    selected_classes = [class_map[name] for name in requested if name in class_map]
    if not selected_classes:
        retained = high_confidence_classes(summary)
        pool = retained or summary.classes
        selected_classes = sorted(
            pool, key=lambda item: item.quality_score, reverse=True
        )[:2]
    candidates: list[EvidenceFrame] = []
    period = 1.0 / summary.analysis_fps

    def add(timestamp: float, reason: str, score: float) -> None:
        candidates.append(
            EvidenceFrame(
                min(summary.end, max(summary.start, timestamp)), reason, score
            )
        )

    for target_time in parsed.timestamps:
        for offset, score in ((-period, 105), (0.0, 110), (period, 105)):
            add(target_time + offset, "question timestamp", score)
    for item in selected_classes:
        if question_type in {"first_visible", "quadrant_first_visible"}:
            for offset, score in ((-period, 100), (0.0, 110), (period, 105)):
                add(
                    item.first_seen + offset,
                    f"{item.class_name} first appearance",
                    score,
                )
        elif question_type in {
            "last_visible",
            "quadrant_last_visible",
            "retrieval_after_time",
        }:
            for offset, score in ((-period, 105), (0.0, 110), (period, 100)):
                add(
                    item.last_seen + offset, f"{item.class_name} last appearance", score
                )
        elif question_type in {
            "first_action_time",
            "ordinal_action_time",
            "all_action_times",
        }:
            for track in (track for track in item.tracks if track.quality.retained):
                for endpoint, label in (
                    (track.first_seen, "appearance transition"),
                    (track.last_seen, "disappearance transition"),
                ):
                    for offset, score in ((-period, 99), (0.0, 108), (period, 99)):
                        add(endpoint + offset, f"{item.class_name} {label}", score)
        elif question_type == "max_simultaneous":
            for value in item.max_count_times[:3]:
                add(value, f"{item.class_name} maximum simultaneous count", 110)
        elif question_type in {"location", "quadrant_at_time"}:
            value = (
                item.first_seen if "first" in question.casefold() else item.last_seen
            )
            add(value, f"{item.class_name} location endpoint", 110)
        elif question_type in {
            "first_disappearance",
            "reentry_exists",
            "reentry_count",
            "visible_duration",
            "frame_percentage",
            "target_only_duration",
            "co_visible_duration",
        }:
            for interval in item.visible_intervals[:4]:
                for value, label in (
                    (interval.start, "visibility starts"),
                    (interval.end, "visibility ends"),
                ):
                    for offset, score in ((-period, 96), (0.0, 106), (period, 96)):
                        add(value + offset, f"{item.class_name} {label}", score)
        else:
            for interval in item.visible_intervals:
                add(interval.start, f"{item.class_name} visibility starts", 95)
                add(interval.end, f"{item.class_name} visibility ends", 94)
            for track in item.tracks:
                for value in track.best_times[:2]:
                    add(value, f"high-confidence {item.class_name}", 90)
    anchor_count = max(2, budget)
    for index in range(anchor_count):
        fraction = index / max(1, anchor_count - 1)
        boundary = index in {0, anchor_count - 1}
        score = 200 if boundary and budget >= 4 else 10
        add(
            summary.start + fraction * (summary.end - summary.start),
            "uniform coverage",
            score,
        )
    candidates.sort(key=lambda item: (-item.score, item.timestamp))
    chosen: list[EvidenceFrame] = []
    tolerance = period * 0.45
    for candidate in candidates:
        if all(
            (abs(candidate.timestamp - item.timestamp) > tolerance for item in chosen)
        ):
            chosen.append(candidate)
        if len(chosen) >= budget:
            break
    return tuple(sorted(chosen, key=lambda item: item.timestamp))
