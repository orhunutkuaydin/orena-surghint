"""Answer a question about one surgical video clip."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

from .weights import METADATA, prepare, references, tensor_inventory, verify_file


def native_qwen_kernels() -> None:
    """Use native PyTorch kernels for the release inference configuration."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    required = (
        "causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule", "is_fast_path_available",
        "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
    )
    if any(not hasattr(module, name) for name in required):
        raise RuntimeError("Unsupported Transformers Qwen kernel interface")
    for name in required[:4]:
        setattr(module, name, None)
    module.is_fast_path_available = False


def validate_model_directory(path: Path) -> None:
    for name in METADATA:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    metadata = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "qwen3_5":
        raise ValueError("The model directory must contain a merged Qwen3.5/3.6 model")
    tensor_inventory(path)


def answer_clip(request: dict, video: Path, weights: Path, model_dir: Path | None = None) -> str:
    import torch  # Initialize before importing video decoders.
    import cv2
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .detector import Detector
    from .prompts import encode
    from .video import open_video, track_video, visuals

    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    cv2.setNumThreads(1)
    native_qwen_kernels()
    print("Loading RF-DETR-L and Qwen3.6-27B.", file=sys.stderr, flush=True)
    detector = Detector(weights / "custom/rfdetr-large.pth")
    model_path = model_dir or weights / "runtime/qwen"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": "cuda:0"},
    ).eval()
    model.config.use_cache = True
    layers = [layer for layer in model.modules() if hasattr(layer, "chunk_gated_delta_rule")]
    if not layers or any(
        layer.causal_conv1d_fn is not None
        or layer.chunk_gated_delta_rule.__name__ != "torch_chunk_gated_delta_rule"
        or layer.recurrent_gated_delta_rule.__name__ != "torch_recurrent_gated_delta_rule"
        for layer in layers
    ):
        raise RuntimeError("Native Qwen kernels were not applied to every linear-attention layer")
    reader = open_video(video)
    tracking = track_video(detector, reader, request)
    evidence, frames = visuals(reader, tracking, request)
    print("Detector and tracking complete; generating the answer.", file=sys.stderr, flush=True)
    batch = encode(processor, request, evidence, frames).to("cuda")
    with torch.inference_mode():
        output = model.generate(**batch, do_sample=False, max_new_tokens=256, use_cache=True)
    answer = processor.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("Empty Qwen answer")
    return answer


def _worker(request: dict, video: Path, weights: Path, model_dir: Path | None, result: Path) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with redirect_stdout(sys.stderr):
        answer = answer_clip(request, video, weights, model_dir)
    write_answer(result, answer)


def run(request: dict, video: Path, weights: Path, model_dir: Path | None = None) -> str:
    with tempfile.TemporaryDirectory(prefix="surghint-") as directory:
        result = Path(directory) / "answer.txt"
        worker = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(request, video, weights, model_dir, result),
        )
        worker.start()
        try:
            worker.join()
        except KeyboardInterrupt:
            worker.terminate()
            worker.join()
            raise
        if worker.exitcode != 0:
            raise RuntimeError(f"Inference worker failed with exit code {worker.exitcode}; see the error log")
        if not result.is_file() or not result.read_text(encoding="utf-8").strip():
            raise RuntimeError("Inference completed without an answer")
        return result.read_text(encoding="utf-8").strip()


def write_answer(path: Path, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".answer-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(answer + "\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Already-trimmed plain clip, approximately 5 fps")
    parser.add_argument("--question", required=True)
    parser.add_argument("--procedure", required=True, help="Surgical procedure name")
    parser.add_argument("--start-time", type=float, required=True, help="Absolute procedure time in seconds")
    parser.add_argument("--end-time", type=float, required=True, help="Absolute procedure time in seconds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--model-dir", type=Path, help="Merged model exported from a custom training run")
    args = parser.parse_args()
    if not all(map(math.isfinite, (args.start_time, args.end_time))) or not 0 <= args.start_time < args.end_time:
        parser.error("Require finite times with 0 <= start-time < end-time")
    if not args.question.strip() or not args.procedure.strip():
        parser.error("Question and procedure must not be empty")
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    video, weights = args.video.resolve(), args.weights_dir.resolve()
    model_dir = args.model_dir.resolve() if args.model_dir else None
    output = args.output.resolve()
    protected = [weights] + ([model_dir] if model_dir else [])
    if output == video or any(
        output.is_relative_to(path) or Path(os.path.abspath(args.output)).is_relative_to(path) for path in protected
    ):
        parser.error("Output must not overwrite the video or a checkpoint")
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Select exactly one CUDA GPU using CUDA_VISIBLE_DEVICES")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("A BF16-capable GPU is required")
        if torch.cuda.get_device_properties(0).total_memory < 55 * 2**30:
            raise RuntimeError("The BF16 27B model needs more GPU memory; an 80 GB GPU is recommended")
        with redirect_stdout(sys.stderr):
            if model_dir:
                validate_model_directory(model_dir)
                verify_file(weights / "custom/rfdetr-large.pth", references()["custom/rfdetr-large.pth"])
            else:
                prepare(weights)
        request = dict(qID="clip", videoID=video.stem, start_time=args.start_time,
                       end_time=args.end_time, procedure_type=args.procedure, question=args.question)
        answer = run(request, video, weights, model_dir)
        write_answer(output, answer)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Inference failed: {exc}\n")
    print(answer)


if __name__ == "__main__":
    main()
