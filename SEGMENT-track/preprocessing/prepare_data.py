"""Prepare training clips and detector tracking evidence from SEGMENT annotations."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from training import config
from training.data import atomic_json, file_digest, fingerprint, pipeline_digest, read_json, window


def seconds(value: str) -> float:
    try:
        hours, minutes, secs = map(float, str(value).split(":"))
    except ValueError as exc:
        raise ValueError(f"Expected an hh:mm:ss timestamp, got {value!r}") from exc
    if not all(map(math.isfinite, (hours, minutes, secs))) or hours < 0 or not 0 <= minutes < 60 or not 0 <= secs < 60:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return 3600 * hours + 60 * minutes + secs


def annotation_file(root: Path) -> Path:
    candidates = [root / "data/segment/train.parquet", root / "train.parquet"]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one SEGMENT train.parquet under {root}")
    return found[0]


def parse_rows(records: list[dict], dataset: str) -> list[dict]:
    rows = []
    for item in records:
        if str(item.get("track", "segment")).casefold() != "segment":
            raise ValueError("Only SEGMENT annotations are supported")
        for name in ("id", "video", "question", "procedure_type", "answer"):
            if item.get(name) is None or not str(item[name]).strip():
                raise ValueError(f"Missing annotation field: {name}")
        request = {
            "qID": str(item["id"]), "videoID": str(item["video"]),
            "start_time": seconds(item["timestamp_start"]), "end_time": seconds(item["timestamp_end"]),
            "procedure_type": str(item["procedure_type"]), "question": str(item["question"]),
        }
        row = {"id": f"{dataset}:train:{request['qID']}", "dataset": dataset,
               "request": request, "answer": str(item["answer"])}
        window(row)
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate question IDs in {dataset}")
    return rows


def source_video(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Video filenames must be relative to the dataset directory")
    candidates = [root / "videos" / relative, root / relative]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one source video {name!r} under {root}")
    return found[0].resolve()


def clip_command(source: Path, destination: Path, start: float, end: float) -> list[str]:
    return [
        config.FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
        "-ss", str(start), "-i", str(source), "-t", str(end - start), "-map", "0:v:0",
        "-map_metadata", "-1", "-vf", r"fps=5,scale=-2:trunc(min(ih\,576)/2)*2,setsar=1",
        "-filter_threads", "1", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-threads", "2", "-g", "25", "-keyint_min", "25",
        "-sc_threshold", "0", "-movflags", "+faststart", str(destination),
    ]


def encode_clip(source: Path, destination: Path, start: float, end: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.mp4")
    try:
        subprocess.run(clip_command(source, temporary, start, end), check=True)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError("FFmpeg produced an empty clip")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_window(row: dict, root: Path, detector, detector_digest: str, pipeline: str, encoder: str) -> dict:
    from surghint.video import open_video, track_video

    identity = window(row)
    key = fingerprint(identity)[:24]
    clip = root / "clips" / f"{key}.mp4"
    tracking = root / "tracking" / f"{key}.json"
    marker = root / "metadata" / f"{key}.json"
    source = source_video(config.DATASETS[row["dataset"]], row["request"]["videoID"])
    stat = source.stat()
    metadata = {
        "window": identity, "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "detector_sha256": detector_digest, "pipeline_sha256": pipeline, "ffmpeg": encoder,
        "preparation_sha256": file_digest(Path(__file__)),
    }
    if marker.is_file() and clip.is_file() and tracking.is_file():
        saved = read_json(marker)
        if (saved["metadata"] == metadata and saved["clip_sha256"] == file_digest(clip)
                and saved["tracking_sha256"] == file_digest(tracking)):
            return saved["files"]
    encode_clip(source, clip, identity[2], identity[3])
    reader = open_video(clip)
    try:
        cache = track_video(detector, reader, row["request"])
    finally:
        del reader
    cache["window"] = identity
    atomic_json(tracking, cache)
    files = {
        "clip_path": clip.relative_to(root).as_posix(), "clip_sha256": file_digest(clip),
        "tracking_path": tracking.relative_to(root).as_posix(), "tracking_sha256": file_digest(tracking),
    }
    atomic_json(marker, {"metadata": metadata, "files": files,
                         "clip_sha256": files["clip_sha256"], "tracking_sha256": files["tracking_sha256"]})
    return files


def main() -> None:
    import torch  # Initialize the shared runtime before importing video decoders.
    import pyarrow.parquet as pq

    from surghint.detector import Detector

    if not shutil.which(config.FFMPEG):
        raise RuntimeError("FFmpeg with libx264 must be installed")
    if not torch.cuda.is_available():
        raise RuntimeError("Detector evidence preparation requires a CUDA GPU")
    rows, annotations = [], {}
    for dataset, directory in config.DATASETS.items():
        path = annotation_file(directory)
        annotations[dataset] = file_digest(path)
        rows.extend(parse_rows(pq.read_table(path).to_pylist(), dataset))
    rows.sort(key=lambda row: (row["dataset"], row["request"]["videoID"],
                              row["request"]["start_time"], row["request"]["end_time"], row["id"]))
    if not rows or (config.REQUIRE_FULL_TRAIN_SPLIT and len(rows) != config.TRAIN_QUESTIONS):
        raise ValueError(f"Expected {config.TRAIN_QUESTIONS} training questions, found {len(rows)}")
    root = config.DATA_DIR.resolve()
    checkpoint = config.WEIGHTS_DIR / "custom/rfdetr-large.pth"
    detector_digest = file_digest(checkpoint)
    detector = Detector(checkpoint)
    pipeline = pipeline_digest()
    encoder = subprocess.run([config.FFMPEG, "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    prepared = {}
    for index, row in enumerate(rows, 1):
        key = fingerprint(window(row))
        if key not in prepared:
            prepared[key] = prepare_window(row, root, detector, detector_digest, pipeline, encoder)
        row.update(prepared[key])
        if index % 100 == 0 or index == len(rows):
            print(f"Prepared {index}/{len(rows)} questions", flush=True)
    atomic_json(root / "manifest.json", {
        "split": "train", "pipeline_sha256": pipeline, "detector_sha256": detector_digest,
        "annotation_sha256": annotations, "rows": rows,
    })
    print(f"Training manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
