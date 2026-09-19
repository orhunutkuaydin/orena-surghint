#!/usr/bin/env python3
"""Single-frame inference for Team VLM_MAXXING's ORena FRAME solution."""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
import re
from pathlib import Path

import torch
import torchvision.transforms.functional as tvf
from PIL import Image

from . import detector as detector_loader
from .hints import raw_from_detections, render_counting_hint, render_hint
from .postprocessing import CLASSES, RAW_INPUT_FLOOR
from .weights import (
    DINOV3_COMMIT,
    prepare_for_inference,
    verify_qwen_runtime,
    verify_source,
)

IMAGE_SIZE = (960, 540)
DETECTOR_SHORT_SIDE = 1024
MAX_NEW_TOKENS = 128
COUNTING_QUESTION = re.compile(r"^\s*How many ", re.IGNORECASE)
DETECTOR_SHA256 = "df29e34ecee1502fb108c29840cf370ddb4b45748552538058d7cba5ff758ae4"

FO_DEFINITIONS = """Foreign Object (FO) Definition
==============================
A foreign object (FO) is any object fully introduced into the patient's body
cavity during surgery that must be retrieved or accounted for. Importantly,
standard surgical instruments that remain connected to the external environment
(e.g., graspers, scissors, trocars, staplers, cameras) are not considered foreign
objects. Furthermore, we exclude detachable parts of surgical instruments,
particularly anvil components of staplers.


Foreign Object Classes
======================

Sponge
------
A soft, absorbent material used to soak up fluids. They are typically white when
fresh and can become reddish-brown when saturated with blood.

Clip
----
A small metal or polymer device used to seal vessels or ducts. May potentially
remain in the body. Clips only count as foreign objects once placed in the
abdomen. They do not count as foreign objects while loaded within the clip
applier instrument.

Specimen Bag
------------
A sterile pouch used to collect and retrieve resected tissue or organs from the
body cavity. Only consider the pouch itself as foreign object and ignore the
string attached to it.

Silicone Loop
-------------
A soft and flexible, typically white band used to encircle and control blood
vessels or structures for isolation and traction.

External Drain
--------------
A clear or fluid-filled tube used to evacuate fluids from the surgical site to
the outside of the body. Its tip may be temporarily visible within the surgical
field during placement or adjustment.

Needle
------
A sharp, pointed metal instrument, straight or curved, used for placing sutures.
If only the string of the needle is visible, it does not count as "the needle is
visible". Only if the needle itself is visible in the video.

Gallstone
---------
Calcified concretion originating from the gallbladder with a roundish,
white/yellow appearance. Gallstones that are within a specimen bag count as
retrieved. Specimen bags are annotated separately.

Specimen
--------
Excised biological tissue (e.g. appendix, resected bowel segment) that must be
retrieved from the body cavity before procedure completion. As soon as every
connection to other body anatomy is cut, the cut off tissue/organ counts as
specimen. This excludes fat or blood. Specimens that are within a specimen bag
count as retrieved. Specimen bags are annotated separately.

Mesh
----
A screen-like patch that surgeons use to reinforce a weak area in the body's
muscle wall. This differs from a sponge which appears more like a tightly woven
cloth. It is an implantable foreign body that is not removed at the end of the
surgery.

Absorbable Hemostatic Agent
---------------------------
A resorbable material applied to a bleeding surface to promote clotting,
intended to be left in the body and absorbed. Typically appears as a white or
pale-yellow frizzy mesh.
"""

SYSTEM_PROMPT = (
    "You are a surgical assistant. You are given an endoscopic image from a "
    "minimally invasive procedure. Analyze the image and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
    + FO_DEFINITIONS
    + "\n\nAnswer with the final answer only, in exactly the format the "
    "question requests: for example a single number, 'yes' or 'no', a class "
    "name, or a comma-separated list of class names. Do not explain your "
    "reasoning and do not add any other text."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required weight: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"invalid weight {path}: sha256 {actual}, expected {expected_sha256}")


def detector_hints(
    cases: list[tuple[Image.Image, str]], weights: Path, device: str
) -> list[str]:
    source = weights / "base/dinov3/source"
    verify_source(source, DINOV3_COMMIT)
    os.environ[detector_loader.DINOV3_SOURCE_ENV] = str(source)
    state = weights / "runtime/detector/serving_state.pth"
    require_file(state, DETECTOR_SHA256)
    wrapper = detector_loader.load_runtime_detector(state, device=device)
    hints = []
    try:
        for image, question in cases:
            width, height = image.size
            scale = DETECTOR_SHORT_SIDE / min(width, height)
            resized = image.resize(
                (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
            )
            tensor = tvf.normalize(
                tvf.to_tensor(resized),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ).to(device)
            with torch.inference_mode():
                output = wrapper([tensor])[0]
            detections = []
            for score, label, box in zip(
                output["scores"].tolist(),
                output["labels"].tolist(),
                output["boxes"].tolist(),
            ):
                if float(score) < RAW_INPUT_FLOOR:
                    continue
                index = int(label)
                if 0 <= index < len(CLASSES):
                    x1, y1, x2, y2 = (float(value) / scale for value in box)
                    detections.append((CLASSES[index], float(score), x1, y1, x2, y2))
            raw = raw_from_detections(detections)
            renderer = (
                render_counting_hint
                if COUNTING_QUESTION.match(question)
                else render_hint
            )
            hints.append(renderer(raw, width=width, height=height))
            logging.info("Detector complete: %d/%d", len(hints), len(cases))
            del tensor, output
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
    return hints


def detector_hint(image: Image.Image, weights: Path, device: str, question: str) -> str:
    return detector_hints([(image, question)], weights, device)[0]


def answers(
    cases: list[tuple[Image.Image, str]], weights: Path, device: str
) -> list[str]:
    if not cases:
        return []
    model_dir = weights / "runtime/qwen"
    verify_qwen_runtime(model_dir)
    logging.info("Running the foreign-object detector")
    hints = detector_hints(cases, weights, device)
    logging.info("Detector complete; loading Qwen3.5-9B")
    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLM
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = AutoVLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    results = []
    try:
        for (image, question), hint in zip(cases, hints, strict=True):
            frame = image.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": f"{hint}\n{question}"},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            if text.endswith("<think>\n"):
                text += "\n</think>\n\n"
            inputs = processor(
                text=[text], images=[frame], padding=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            trimmed = generated[:, inputs["input_ids"].shape[1] :]
            result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            if not result.strip():
                raise RuntimeError("Qwen returned an empty answer")
            results.append(result)
            logging.info("VLM complete: %d/%d", len(results), len(cases))
            del inputs, generated, trimmed
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    return results


def answer(image: Image.Image, question: str, weights: Path, device: str) -> str:
    return answers([(image, question)], weights, device)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="path to a surgical frame")
    parser.add_argument("--question", required=True, help="question about the frame")
    parser.add_argument("--output", required=True, type=Path, help="answer text file")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.question.strip():
        raise ValueError("question must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    with Image.open(args.image) as handle:
        handle.verify()
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
    weights = args.weights_dir.resolve()
    prepare_for_inference(weights)
    result = answer(image, args.question, weights, "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(result, end="" if result.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
