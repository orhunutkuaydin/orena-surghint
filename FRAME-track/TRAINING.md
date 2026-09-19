# SurgHint · FRAME training

[← Inference guide](README.md#inference) · [Checkpoint licences](https://huggingface.co/OUAydin/ORena-SurgHint-solution)

Run these commands from `FRAME-track/` in the environment described in the [inference guide](README.md#1-install).

## Scope and required artifacts

Detector annotations for **HeiCo** and **Surgical Gauze** are available in the [SurgHint annotation archive](https://zenodo.org/records/22769709).

## Base checkpoints

Download the pinned Qwen base:

```bash
python -m surghint.weights download-qwen
```

Detector training uses the frozen **DINOv3 7B (ViT-7B/16)** backbone and requires these two checkpoints from [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/):

| Original checkpoint filename | Local path |
| --- | --- |
| `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` (ViT-7B/16 backbone) | `weights/base/dinov3/dinov3_vit7b16_backbone.pth` |
| `dinov3_vit7b16_coco2017_detr_head-b0235ff7.pth` (COCO2017 head) | `weights/base/dinov3/dinov3_vit7b16_detector.pth` |

Fetch the pinned source trees for DINOv3 7B and Plain-DETR:

```bash
python -m surghint.weights sources --training
```

## Detector training on HeiCo

Download the [SurgHint annotations](https://doi.org/10.5281/zenodo.22769709) and obtain videos through the [ORena FOCUS data pages](https://orena-focus-challenge.org/data/) or [HeiCo dataset page](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa), under their access terms.

Extract the annotations under `data/`; `HEICO` below is your video directory. Convert and validate:

```bash
python -m preprocessing.prepare_data coco \
  --annotations data/surghint_annotations/heico/annotations.jsonl \
  --source-map data/surghint_annotations/heico/id_to_source.csv \
  --videos HEICO \
  --output data/surghint-heico

python -m training.train_detector \
  --data data/surghint-heico \
  --output outputs/heico-detector \
  --dry-run
```

`--dry-run` validates the dataset. Train with:

```bash
python -m training.train_detector \
  --data data/surghint-heico \
  --output outputs/heico-detector
```

The defaults are a 1024-pixel short side, batch size 2, gradient accumulation 8, AdamW with learning rate `2e-5`, and 12 epochs. DINOv3 7B remains frozen in BF16; the eight-class head is trained in FP32 and saved to `outputs/heico-detector/last_head.pth`.

## VLM training

Metadata preparation requires the gated [HeiCo](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and [LapChole](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) FRAME datasets; `HEICO` and `LAPCHOLE` are their local directories.

```bash
python -m preprocessing.prepare_data metadata \
  --heico HEICO \
  --lapchole LAPCHOLE \
  --output data \
  --extract
```

Each dataset directory must contain `train.parquet` and `test.parquet`, either at its root or under `data/frame/`, and the corresponding videos at its root or under `videos/`. Preparation requires 13,748 train and 6,252 test questions, extracts 15,212 center frames and their adjacent frames, and writes metadata under `data/metadata/`.

Both splits are used for training; `test.parquet` is not held out.

Prepare detector hints from the original out-of-fold predictions, then train the VLM:

```bash
python -m preprocessing.prepare_data hints \
  --boxes boxes_threshold_005.parquet \
  --manifest frame_manifest.parquet \
  --dimensions frame_dimensions.parquet \
  --output data/metadata/hints.parquet \
  --strict

python -m training.train_vlm \
  --data data \
  --hints data/metadata/hints.parquet \
  --output outputs/qwen-lora
```

| Table | Required columns |
| --- | --- |
| Boxes | `frame_id`, `dataset`, `video`, `split`, `fold`, `source`, `threshold`, `class_name`, `confidence`, `x1`, `y1`, `x2`, `y2`, `width`, `height` |
| Frame manifest | `frame_id`, `dataset`, `video`, `split`, `fold` |
| Frame dimensions | `frame_id`, `width`, `height` |

Box coordinates are in original-image pixels, with a confidence floor of 0.05. The manifest has one row per frame and fold IDs 0–3. The dimensions table includes frames with no detections. Both strict preparation and `train_vlm` enforce the original rendered-hint digest.

| Setting | Default |
| --- | --- |
| LoRA targets | Language attention/MLP and vision linear layers |
| LoRA parameters | Rank 16; alpha 32; dropout 0.05 |
| Detector evidence | No hint dropout |
| Optimizer schedule | Learning rate `5e-5`; three-epoch cosine schedule; stop at optimizer step 1750 |
| Batch | Size 4; gradient accumulation 4 |
| Images | 960×540; adjacent-frame and appearance augmentation |
| Reproducibility | Seed 0; gradient checkpointing |
| Saved adapter | `outputs/qwen-lora/adapter/` |

Smoke test (batch size 1, three optimizer steps):

```bash
python -m training.train_vlm \
  --data data \
  --hints data/metadata/hints.parquet \
  --output outputs/vlm-smoke \
  --smoke-steps 3
```

This requires the full validated metadata and hints. It checks finite losses and changed adapter tensors, then writes an adapter and `smoke_report.json`.
