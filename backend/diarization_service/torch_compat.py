from __future__ import annotations

import inspect


def ensure_torch_amp_compatibility() -> None:
    try:
        import torch
    except Exception:
        return

    has_compatible_fwd = _accepts_device_type(getattr(torch.amp, "custom_fwd", None))
    has_compatible_bwd = _accepts_device_type(getattr(torch.amp, "custom_bwd", None))
    if has_compatible_fwd and has_compatible_bwd:
        return

    cuda_amp = getattr(torch.cuda, "amp", None)
    if cuda_amp is None:
        return

    if not has_compatible_fwd and hasattr(cuda_amp, "custom_fwd"):
        torch.amp.custom_fwd = _device_type_compatible_custom_fwd(cuda_amp.custom_fwd)  # type: ignore[attr-defined]

    if not has_compatible_bwd and hasattr(cuda_amp, "custom_bwd"):
        torch.amp.custom_bwd = _device_type_compatible_custom_bwd(cuda_amp.custom_bwd)  # type: ignore[attr-defined]


def _accepts_device_type(function: object | None) -> bool:
    if function is None:
        return False
    try:
        return "device_type" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _device_type_compatible_custom_fwd(cuda_custom_fwd):
    def custom_fwd(fwd=None, *, cast_inputs=None, device_type=None):
        del device_type
        return cuda_custom_fwd(fwd=fwd, cast_inputs=cast_inputs)

    return custom_fwd


def _device_type_compatible_custom_bwd(cuda_custom_bwd):
    def custom_bwd(bwd=None, *, device_type=None):
        del device_type
        return cuda_custom_bwd(bwd=bwd)

    return custom_bwd
