"""Memory-efficient loading for the DINOv3 ViT-7B/16 Plain-DETR model."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .detr_utils import (
    enable_bf16_frozen_backbone,
    replace_class_heads,
)
from .postprocessing import CLASSES

HUB_ENTRY = "dinov3_vit7b16_de"
DINOV3_SOURCE_ENV = "ORENA_DINOV3_SOURCE"
# Local filenames for the two upstream checkpoints.
BACKBONE_WEIGHTS = "dinov3_vit7b16_backbone.pth"
DETECTOR_WEIGHTS = "dinov3_vit7b16_detector.pth"
EXPECTED_HEAD_EPOCH = 10
SERVING_STATE_NAME = "serving_state.pth"
LOG = logging.getLogger(__name__)


def _hub_load(**kwargs: Any) -> Any:
    import torch

    source = os.environ.get(DINOV3_SOURCE_ENV)
    if source:
        path = Path(source).resolve()
        if not (path / "hubconf.py").is_file():
            raise FileNotFoundError(f"invalid DINOv3 source tree: {path}")
        return torch.hub.load(str(path), HUB_ENTRY, source="local", **kwargs)
    raise RuntimeError("DINOv3 source directory is not configured; run python -m surghint.weights sources")


def _swap_class_heads(wrapper: Any) -> list[str]:
    """Replace the upstream heads with the eight FRAME classes."""

    return replace_class_heads(wrapper, num_classes=len(CLASSES))


@contextlib.contextmanager
def _default_dtype(dtype: Any):
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def _mmap_torch_load():
    """Memory-map hub checkpoints, falling back to ordinary loading if needed.

    Memory mapping reduces resident CPU memory without changing tensor values.
    The original loader is restored when the context exits.
    """

    import torch

    original = torch.load

    def load_with_mmap(*args: Any, **kwargs: Any) -> Any:
        if "mmap" not in kwargs:
            try:
                return original(*args, mmap=True, **kwargs)
            except (RuntimeError, TypeError, ValueError):
                pass
        return original(*args, **kwargs)

    torch.load = load_with_mmap
    try:
        yield
    finally:
        torch.load = original


def _seed_hub_checkpoint_cache(models_dir: Path) -> None:
    """Symlink local checkpoints into the cache used by DINOv3's hub loader.

    This avoids copying the checkpoints when the loader resolves file URLs.
    """

    import torch

    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for name in (BACKBONE_WEIGHTS, DETECTOR_WEIGHTS):
        source = (models_dir / name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = checkpoints / name
        if target.resolve() != source:
            target.unlink(missing_ok=True)
            target.symlink_to(source)


def build_architecture() -> Any:
    """Construct the eight-class detector on the meta device.

    Runtime loading supplies parameter and buffer storage with ``assign=True``.
    """

    import torch

    with _default_dtype(torch.bfloat16), torch.device("meta"):
        wrapper = _hub_load(pretrained=False)
    _swap_class_heads(wrapper)
    return wrapper


def assert_fully_materialized(root: Any) -> None:
    """Reject parameters or buffers left on the meta device before transfer."""

    remaining = [
        f"parameter:{name}"
        for name, tensor in root.named_parameters()
        if tensor.is_meta
    ]
    remaining.extend(
        f"buffer:{name}"
        for name, tensor in root.named_buffers()
        if tensor.is_meta
    )
    if remaining:
        raise RuntimeError(
            "serving state left detector tensors on the meta device: "
            f"{remaining[:8]}"
        )


def load_runtime_detector(
    serving_state: Path,
    device: str | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Load the serving tensors into the detector architecture.

    ``progress`` reports construction, state loading, and device transfer.
    """

    import torch

    started = time.monotonic()

    def note(message: str) -> None:
        detail = f"{message} ({time.monotonic() - started:.1f}s elapsed)"
        LOG.info("Detector load: %s", detail)
        if progress is not None:
            progress(detail)

    note("building weight-free architecture")
    wrapper = build_architecture()
    note("architecture built; memory-mapping serving state")
    state = torch.load(serving_state, map_location="cpu", mmap=True, weights_only=True)
    note("serving state mapped; assigning tensors")
    wrapper.load_state_dict(state, strict=True, assign=True)
    note("serving tensors assigned")
    # Release the state-dict references before transferring the model to the GPU.
    del state
    gc.collect()
    note("state-dict references released")
    assert_fully_materialized(wrapper)
    note("all detector tensors materialized")
    # assign=True preserves requires_grad; the BF16 adapter requires a frozen backbone.
    backbone_trainable = [
        name
        for name, param in wrapper.detector.backbone.named_parameters()
        if param.requires_grad
    ]
    if backbone_trainable:
        raise RuntimeError(
            f"constructed backbone has trainable tensors: {backbone_trainable[:4]}"
        )
    if device is not None:
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        note(f"moving detector to {device}")
        wrapper = wrapper.to(device)
        note(f"detector moved to {device}")
    enable_bf16_frozen_backbone(wrapper.detector, dtype=torch.bfloat16)
    note("BF16 backbone adapter installed")
    wrapper.eval()
    note("detector ready")
    return wrapper
