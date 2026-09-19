"""Eight-class heads and mixed-precision backbone support for Plain-DETR."""

from __future__ import annotations

from typing import Any


def replace_class_heads(
    root: Any,
    *,
    num_classes: int,
    source_classes: int = 91,
) -> list[str]:
    """Replace class linear layers while preserving aliases, device, and dtype."""

    import math

    import torch.nn as nn

    replacements: dict[int, nn.Linear] = {}
    swapped: list[str] = []
    bias_prior = -math.log((1.0 - 0.01) / 0.01)

    def replacement_for(linear: nn.Linear) -> nn.Linear:
        key = id(linear)
        if key not in replacements:
            new = nn.Linear(
                linear.in_features,
                num_classes,
                bias=linear.bias is not None,
            ).to(device=linear.weight.device, dtype=linear.weight.dtype)
            nn.init.normal_(new.weight, std=0.01)
            new.weight.requires_grad_(linear.weight.requires_grad)
            if new.bias is not None:
                nn.init.constant_(new.bias, bias_prior)
                assert linear.bias is not None
                new.bias.requires_grad_(linear.bias.requires_grad)
            replacements[key] = new
        return replacements[key]

    # The decoder and transformer share this ModuleList. Replacing its children
    # preserves their aliases without duplicating the class heads.
    for module_name, module in root.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for index, child in enumerate(list(module)):
            if isinstance(child, nn.Linear) and child.out_features == source_classes:
                module[index] = replacement_for(child)
                swapped.append(f"{module_name}[{index}]")
    if not swapped:
        raise RuntimeError(f"no {source_classes}-way class Linears found")
    return swapped


def enable_bf16_frozen_backbone(detector: Any, *, dtype: Any = None) -> bool:
    """Run the frozen backbone in BF16 and return FP32 feature maps.

    The FP32 boundary precedes the input projection, transformer, and prediction
    heads, preserving precision for absolute-pixel box regression. ``dtype``
    defaults to BF16 and may also be FP16.

    Return True on installation, or False if the same adapter is installed.
    """

    import types

    import torch

    dtype = torch.bfloat16 if dtype is None else dtype
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError(f"unsupported frozen-backbone dtype: {dtype}")
    backbone = detector.backbone
    if getattr(backbone, "_bf16_frozen_fp32_output", False):
        installed = getattr(backbone, "_frozen_backbone_compute_dtype", None)
        if installed is not None and installed != dtype:
            raise RuntimeError(
                f"frozen-backbone adapter already uses {installed}, requested {dtype}"
            )
        return False
    trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(
            "bf16 backbone isolation requires a completely frozen backbone; "
            f"trainable tensors include {trainable[:5]}"
        )
    # Store frozen parameters in the compute dtype; keep the trainable head FP32.
    backbone.to(dtype=dtype)
    original_forward = backbone.forward

    def forward_with_fp32_output(self, tensor_list):
        tensors = tensor_list.tensors
        device_type = tensors.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=device_type == "cuda",
            cache_enabled=False,
        ):
            features, positions = original_forward(tensor_list)

        fp32_features = []
        for feature in features:
            if hasattr(feature, "tensors") and hasattr(feature, "mask"):
                fp32_features.append(type(feature)(feature.tensors.float(), feature.mask))
            elif torch.is_tensor(feature):
                fp32_features.append(feature.float())
            else:
                raise TypeError(f"unsupported backbone feature type: {type(feature)!r}")
        fp32_positions = [position.float() for position in positions]
        return fp32_features, fp32_positions

    backbone.forward = types.MethodType(forward_with_fp32_output, backbone)
    backbone._bf16_frozen_fp32_output = True
    backbone._frozen_backbone_compute_dtype = dtype
    return True
