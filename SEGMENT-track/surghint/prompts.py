"""Surgical question-answering prompts and absolute frame labels."""
from types import SimpleNamespace

import torch
from PIL import Image

from .hints import CONFIG

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

SYSTEM_PROMPT_PREFIX = "You are a surgical assistant. You are given frames sampled in temporal order from an endoscopic video clip of a minimally invasive procedure. Analyze the frames and answer the surgical question based on the visual evidence, paying attention to motion and to changes between frames. Be precise and concise.\n\n"
SYSTEM_PROMPT_SUFFIX = "\n\nTimestamps shown on the frames and in the frame labels are the elapsed procedure time; answer time questions with that absolute time. Answer with the final answer only, in exactly the format the question requests: for example a single number, 'yes' or 'no', a class name, a comma-separated list of class names, a percentage, or one or more timestamps as hh:mm:ss. Do not explain your reasoning and do not add any other text."
SYSTEM_NOTE = "\n\nAdditional detector tracks are uncertain visual evidence. Verify them against the frames. Track identities are local to this clip and can split or switch. Detection boundaries are visibility estimates, not proof of an action. Use the absolute time labels for timestamp answers. Box legend: SP=Sponge; CL=Clip; SB=Specimen Bag; SL=Silicone Loop; ED=External Drain; NE=Needle; GA=Gallstone; SE=Specimen."


def seconds_to_timestamp_precise(seconds):
    tenths = max(0, int(round(float(seconds) * 10)))
    whole, decimal = divmod(tenths, 10)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{decimal}"


def format_frame_label(frame_number, timestamp, clip_start):
    if frame_number < 1 or timestamp < clip_start:
        raise ValueError("Frame number and timestamp must lie inside the clip")
    return f"Frame {frame_number:02d} | video time {seconds_to_timestamp_precise(timestamp)} | clip offset {seconds_to_timestamp_precise(timestamp - clip_start)}"


def build_segment_conversation(
    images, timestamps, request, system_prompt, context_note
):
    if not images or len(images) != len(timestamps):
        raise ValueError("One timestamp is required for every non-empty frame")
    content = [
        {
            "type": "text",
            "text": f"Video clip of the procedure: {request.procedure_type}.",
        }
    ]
    for position, (image, timestamp) in enumerate(
        zip(images, timestamps, strict=True), 1
    ):
        content.append(
            {
                "type": "text",
                "text": format_frame_label(
                    position, timestamp, float(request.start_time)
                ),
            }
        )
        content.append({"type": "image", "image": image})
    if context_note:
        content.append({"type": "text", "text": context_note})
    content.append({"type": "text", "text": request.question})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def encode(processor, request, evidence, frames):
    images = [Image.fromarray(frame) for frame in frames]
    conversation = build_segment_conversation(
        images,
        evidence["times"],
        SimpleNamespace(**request),
        SYSTEM_PROMPT_PREFIX + FO_DEFINITIONS + SYSTEM_PROMPT_SUFFIX + SYSTEM_NOTE,
        evidence["hint"],
    )
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("Processor did not honor the answer-only chat template")
    batch = processor(
        text=[prompt], images=images, do_resize=False, return_tensors="pt"
    )
    grid = batch["image_grid_thw"]
    if len(grid) != CONFIG.frames or not torch.all(grid[:, 0] == 1):
        raise RuntimeError("Processor changed image count")
    if not torch.all(grid[:, 1] == CONFIG.height // 16) or not torch.all(
        grid[:, 2] == CONFIG.width // 16
    ):
        raise RuntimeError("Processor changed the explicit image resolution")
    if batch["input_ids"].shape[1] > 24576:
        raise RuntimeError("Input exceeds token budget; no truncation")
    return batch
