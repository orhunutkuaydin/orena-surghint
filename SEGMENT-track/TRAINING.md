# SurgHint · SEGMENT training

[← Inference guide](README.md#inference) · [Checkpoint licences](https://huggingface.co/OUAydin/ORena-SurgHint-solution)

Run these commands from `SEGMENT-track/` in the environment described in the [inference guide](README.md#1-install).

Configure paths and training settings in [training/config.py](training/config.py), which also pins the dataset revisions.

## Prepare the data

Obtain the [HeiCo](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and [LapChole](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) SEGMENT annotations and original videos under their access terms:

```text
data/
├── heico/
│   ├── data/segment/train.parquet
│   └── videos/
└── lapchole/
    ├── data/segment/train.parquet
    └── videos/
```

First [download the released detector and adapter files](README.md#2-download-the-released-checkpoints). Preparation requires FFmpeg with libx264 and a CUDA GPU. It uses the 13,746-question training split; `REQUIRE_FULL_TRAIN_SPLIT = False` permits smaller datasets.

```bash
python -m pip install -e '.[training]' --extra-index-url https://download.pytorch.org/whl/cu128
python -m preprocessing.prepare_data
python -m training.data
```

Preparation creates 5-fps clips per unique question window, computes detections and tracks, and writes `data/segment/manifest.json` with file hashes. Reference answers are stored separately from detector evidence and excluded from input prompts.

## Train the VLM

Two-GPU data-parallel training was run on H200s; each GPU holds the full BF16 model. Install the CUDA toolkit and the [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) kernels:

```bash
python -m pip install --no-build-isolation 'flash-attn>=2,<3' causal-conv1d flash-linear-attention
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 -m training.train_vlm
```

The script downloads the pinned base model and applies loss only to answer tokens, using the inference frame selection, rendering and prompt.

| Setting | Value |
| --- | --- |
| LoRA | Language attention and MLP projections; rank 16, alpha 32, dropout 0.05 |
| Batch size | 1 per GPU, gradient accumulation 4; effective batch 8 |
| Optimizer | Fused AdamW, learning rate `1e-4`, weight decay 0.01 |
| Schedule | Two-epoch cosine schedule; training stops after the first epoch |
| Full training split | 1,719 optimizer steps; 3,438-step schedule; 103 warmup steps |
| Inputs | 24 frames at 768×448; no answer truncation |
| Seed | 0 |

Adapters are saved to `outputs/segment-vlm/adapter/`. Resume with `RESUME_CHECKPOINT` in `training/config.py`. Test questions are not used for training or checkpoint selection.

## Export and use a trained VLM

```bash
python -m training.export_model
export CUDA_VISIBLE_DEVICES=0
surghint \
  --model-dir outputs/segment-vlm/model \
  --video clip.mp4 \
  --question "Which foreign object classes are visible?" \
  --procedure "Sigmoid Resection" \
  --start-time 205 \
  --end-time 244 \
  --output answer.txt
```

`--model-dir` loads the custom merged VLM; the released detector is still used. Export runs on CPU and refuses to overwrite existing model exports.

## Train the detector

Set `DETECTOR_DATA` to an eight-class COCO directory containing the images and `_annotations.coco.json`. Download Roboflow's [rf-detr-large-2026.pth](https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth) to `weights/base/rf-detr-large-2026.pth`, the default `DETECTOR_PRETRAINED` path. [Upstream RF-DETR training guide](https://rfdetr.roboflow.com/learn/train/).

```bash
export CUDA_VISIBLE_DEVICES=0
python -m training.train_detector
```

The recipe uses 896-pixel inputs, batch size 4, gradient accumulation 4, 20 epochs, and the final EMA checkpoint. Output is `outputs/segment-detector/training/last_ema.pth`. RF-DETR's required validation loader reuses up to 64 training images; it is not a held-out evaluation set and does not select the checkpoint.

Detector annotations for **HeiCo** and **Surgical Gauze** are available in the [SurgHint annotation archive](https://zenodo.org/records/22769709).
