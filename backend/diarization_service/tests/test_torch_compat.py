from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from backend.diarization_service.torch_compat import ensure_torch_amp_compatibility


class TorchAmpCompatTest(unittest.TestCase):
    def test_wraps_cuda_amp_decorators_to_accept_device_type(self) -> None:
        def cuda_custom_fwd(fwd=None, *, cast_inputs=None):
            def decorator(function):
                function._cast_inputs = cast_inputs
                return function

            return decorator if fwd is None else decorator(fwd)

        def cuda_custom_bwd(bwd=None):
            def decorator(function):
                function._wrapped_bwd = True
                return function

            return decorator if bwd is None else decorator(bwd)

        fake_torch = types.SimpleNamespace(
            amp=types.SimpleNamespace(
                custom_fwd=lambda fwd=None, *, cast_inputs=None: fwd,
                custom_bwd=lambda bwd=None: bwd,
            ),
            cuda=types.SimpleNamespace(
                amp=types.SimpleNamespace(
                    custom_fwd=cuda_custom_fwd,
                    custom_bwd=cuda_custom_bwd,
                ),
            ),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            ensure_torch_amp_compatibility()

        @fake_torch.amp.custom_fwd(device_type="cuda", cast_inputs="float16")
        def forward(value):
            return value

        @fake_torch.amp.custom_bwd(device_type="cuda")
        def backward(value):
            return value

        self.assertEqual(forward("ok"), "ok")
        self.assertEqual(backward("ok"), "ok")
        self.assertEqual(forward._cast_inputs, "float16")
        self.assertTrue(backward._wrapped_bwd)
