"""ADA audio-mode tests (AUGMENTATION_SPEC.md §B, §D.3).

Importing non_leaking JIT-builds the op/ CUDA extension; the module is skipped
where that build is unavailable (the augment functions themselves are pure torch
and run on CPU).
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    from non_leaking import augment_audio, random_cutout, random_time_translate
except Exception as exc:  # op/ JIT build failure
    pytest.skip(f"non_leaking import failed: {exc}", allow_module_level=True)

SHAPE = (8, 1, 128, 128)


def test_identity_at_p0():
    img = torch.randn(*SHAPE)
    out, mats = augment_audio(img, 0.0)
    assert torch.equal(out, img)
    assert mats == (None, None)


def test_shapes_and_finiteness_at_p1():
    img = torch.randn(*SHAPE)
    out, _ = augment_audio(img, 1.0)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_translation_is_time_axis_only():
    # Encode the row (frequency bin) into the value: any vertical motion would
    # put a foreign row id somewhere.
    img = (
        torch.arange(128).view(1, 1, 128, 1) * 1000.0
        + torch.arange(128).view(1, 1, 1, 128)
    ).expand(*SHAPE).contiguous().float()
    out = random_time_translate(img, 1.0, 0.125)
    assert torch.equal(out.div(1000.0).floor(), img.div(1000.0).floor())


def test_translation_integer_shift_with_replicate_fill():
    ramp = torch.arange(128.0).view(1, 1, 1, 128).expand(*SHAPE).contiguous()
    out = random_time_translate(ramp, 1.0, 0.125)
    for b in range(SHAPE[0]):
        # |shift| <= 16, so the center pixel is never clamped: recover the shift
        # there, then the whole row must be the clamped (replicate-fill) ramp.
        shift = int(64 - out[b, 0, 0, 64].item())
        assert abs(shift) <= round(0.125 * 128)
        expected = (torch.arange(128) - shift).clamp(0, 127).float()
        assert torch.equal(out[b, 0, 0], expected)
        # frequency rows all see the identical shift
        assert torch.equal(out[b, 0], expected.expand(128, 128))


def test_cutout_is_single_zero_rectangle():
    img = torch.ones(*SHAPE)
    out = random_cutout(img, 1.0, 0.4)
    max_side = int(0.4 * 128) + 1
    for b in range(SHAPE[0]):
        zero = out[b, 0] == 0
        assert zero.any()
        rows = zero.any(dim=1).nonzero().flatten()
        cols = zero.any(dim=0).nonzero().flatten()
        assert rows.numel() <= max_side and cols.numel() <= max_side
        # contiguous spans and exact rectangle (single cutout, zero fill)
        assert rows.numel() == rows.max() - rows.min() + 1
        assert cols.numel() == cols.max() - cols.min() + 1
        assert zero.sum() == rows.numel() * cols.numel()
        assert (out[b, 0][~zero] == 1).all()


def test_gradients_flow_for_generator_path():
    img = torch.randn(*SHAPE, requires_grad=True)
    out, _ = augment_audio(img, 1.0)
    out.sum().backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all()
    assert img.grad.abs().sum() > 0


def test_deterministic_under_manual_seed():
    img = torch.randn(*SHAPE)
    torch.manual_seed(1234)
    a, _ = augment_audio(img, 0.7)
    torch.manual_seed(1234)
    b, _ = augment_audio(img, 0.7)
    assert torch.equal(a, b)
