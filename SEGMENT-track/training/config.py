"""Paths and settings for SEGMENT data preparation and training."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"heico": ROOT / "data/heico", "lapchole": ROOT / "data/lapchole"}
DATA_DIR = ROOT / "data/segment"
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs/segment-vlm"
ADAPTER_DIR = OUTPUT_DIR / "adapter"
EXPORT_DIR = OUTPUT_DIR / "model"
RESUME_CHECKPOINT: Path | None = None
FFMPEG = "ffmpeg"
NUM_WORKERS = 4

# Set to False for experiments on a smaller training set.
REQUIRE_FULL_TRAIN_SPLIT = True
TRAIN_QUESTIONS = 13_746
ANNOTATION_REVISIONS = {
    "heico": "4ee0e4b39ee59006b773beec501bb47e251827eb",
    "lapchole": "5b3510cde3ba1135c56c4b4b25b50c4f948e235b",
}

WORLD_SIZE = 2
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 0
MAX_TOKENS = 24_576
ATTENTION_BACKEND = "flash_attention_2"

DETECTOR_DATA = ROOT / "data/detector"
DETECTOR_PRETRAINED = WEIGHTS_DIR / "base/rf-detr-large-2026.pth"
DETECTOR_OUTPUT = ROOT / "outputs/segment-detector"
