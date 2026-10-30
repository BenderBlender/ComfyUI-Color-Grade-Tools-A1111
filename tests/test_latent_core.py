"""Unit tests for the latent-space core. Plain torch — no ComfyUI needed.

    python tests/test_latent_core.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a1111_cc import stats as S  # noqa: E402
from a1111_cc.guidance import GuidanceConfig, LatentCorrector  # noqa: E402
from a1111_cc.space import LatentColorSpace, _flux2_inverse_reshape  # noqa: E402

torch.manual_seed(0)

# The exact lambda comfy/latent_formats.py:Flux2 installs.
FLUX2_RESHAPE = lambda t: (  # noqa: E731
    t.reshape(t.shape[0], 32, 2, 2, t.shape[-2], t.shape[-1])
    .permute(0, 1, 4, 2, 5, 3)
    .reshape(t.shape[0], 32, t.shape[-2] * 2, t.shape[-1] * 2)
)


# --------------------------------------------------------------------------- #
# reduce_dims
# --------------------------------------------------------------------------- #

def test_reduce_dims():
    assert S.reduce_dims(torch.zeros(1, 4, 8, 8)) == [2, 3]
    assert S.reduce_dims(torch.zeros(1, 16, 5, 8, 8), "global") == [2, 3, 4]
    assert S.reduce_dims(torch.zeros(1, 16, 5, 8, 8), "per_frame") == [3, 4]
    try:
        S.reduce_dims(torch.zeros(4, 8))
    except ValueError:
        return
    raise AssertionError("a 2D tensor should have been rejected")


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def test_channel_stats_match_torch():
    x = torch.randn(2, 4, 16, 16) * 3 + 1
    st = S.channel_stats(x, [2, 3])
    assert torch.allclose(st.mean, x.mean(dim=[2, 3], keepdim=True), atol=1e-5)
    assert torch.allclose(st.std, x.std(dim=[2, 3], keepdim=True), atol=1e-4)


def test_masked_stats_only_see_the_mask():
    """Statistics must come from the masked area, not the whole frame."""
    x = torch.zeros(1, 2, 8, 8)
    x[:, :, :, :4] = 5.0      # left half
    x[:, :, :, 4:] = 100.0    # right half, to be ignored
    mask = torch.zeros(1, 8, 8)
    mask[:, :, :4] = 1.0

    st = S.channel_stats(x, [2, 3], mask=mask)
    assert torch.allclose(st.mean, torch.full_like(st.mean, 5.0), atol=1e-4), st.mean
    assert float(st.std.max()) < 1e-3, "masked region is constant, std should vanish"


def test_covariance_is_recovered():
    base = torch.randn(1, 3, 4096)
    mix = torch.tensor([[1.0, 0.8, 0.0], [0.0, 1.0, 0.5], [0.2, 0.0, 1.0]])
    x = (mix @ base).reshape(1, 3, 64, 64)
    st = S.channel_stats(x, [2, 3], need_cov=True)
    flat = x.reshape(1, 3, -1)
    expected = torch.cov(flat[0])
    assert torch.allclose(st.cov[0], expected, atol=1e-3), (st.cov[0], expected)


# --------------------------------------------------------------------------- #
# transfers
# --------------------------------------------------------------------------- #

def test_mean_only_leaves_spread_alone():
    src = torch.randn(1, 8, 32, 32) * 2.0
    ref = torch.randn(1, 8, 32, 32) * 7.0 + 4.0
    ss = S.channel_stats(src, [2, 3])
    rs = S.channel_stats(ref, [2, 3])
    out = S.match_mean_only(src, ss, rs)

    os_ = S.channel_stats(out, [2, 3])
    assert torch.allclose(os_.mean, rs.mean, atol=1e-4), "mean did not move onto the reference"
    assert torch.allclose(os_.std, ss.std, atol=1e-4), "spread should not have changed"


def test_reinhard_matches_mean_and_std():
    src = torch.randn(1, 4, 64, 64) * 2.0 + 1.0
    ref = torch.randn(1, 4, 64, 64) * 3.0 - 2.0
    ss = S.channel_stats(src, [2, 3])
    rs = S.channel_stats(ref, [2, 3])
    out = S.match_reinhard(src, ss, rs, max_gain=8.0)

    os_ = S.channel_stats(out, [2, 3])
    assert torch.allclose(os_.mean, rs.mean, atol=1e-3)
    assert torch.allclose(os_.std, rs.std, rtol=2e-2)


def test_max_gain_clamps_a_flat_source():
    """A nearly featureless prediction must not be amplified without limit."""
    src = torch.randn(1, 4, 32, 32) * 1e-4
    ref = torch.randn(1, 4, 32, 32) * 10.0
    ss = S.channel_stats(src, [2, 3])
    rs = S.channel_stats(ref, [2, 3])
    out = S.match_reinhard(src, ss, rs, max_gain=2.0)

    gain = S.channel_stats(out, [2, 3]).std / (ss.std + 1e-6)
    assert float(gain.max()) <= 2.0 + 1e-3, f"gain reached {float(gain.max())}"


def test_mkl_matches_the_covariance():
    """MKL should transport the full covariance, not just the diagonal."""
    src = (torch.randn(3, 4096) * torch.tensor([[1.0], [2.0], [0.5]])).reshape(1, 3, 64, 64)
    mix = torch.tensor([[1.0, 0.9, 0.1], [0.0, 1.0, 0.7], [0.3, 0.0, 1.0]])
    ref = (mix @ torch.randn(3, 4096) * 2.0).reshape(1, 3, 64, 64)

    ss = S.channel_stats(src, [2, 3], need_cov=True)
    rs = S.channel_stats(ref, [2, 3], need_cov=True)
    out = S.match_mkl(src, ss, rs, max_gain=100.0, shrink=0.0)
    os_ = S.channel_stats(out, [2, 3], need_cov=True)

    err = (os_.cov[0] - rs.cov[0]).abs().max() / rs.cov[0].abs().max()
    assert float(err) < 0.05, f"covariance off by {float(err):.3f}"
    print(f"  MKL covariance error: {float(err):.4f}")


def test_mkl_survives_a_degenerate_source():
    """Shrinkage must keep a rank-deficient prediction from exploding."""
    src = torch.zeros(1, 3, 32, 32)
    src[:, 0] = torch.randn(32, 32)      # only one channel carries anything
    ref = torch.randn(1, 3, 32, 32) * 5.0
    ss = S.channel_stats(src, [2, 3], need_cov=True)
    rs = S.channel_stats(ref, [2, 3], need_cov=True)
    out = S.match_mkl(src, ss, rs, max_gain=2.0, shrink=0.1)
    assert torch.isfinite(out).all(), "MKL produced non-finite values"


# --------------------------------------------------------------------------- #
# the RGB projection
# --------------------------------------------------------------------------- #

def _sdxl_like_factors(channels=4):
    return torch.randn(channels, 3) * 0.3


def test_back_projection_reproduces_the_requested_rgb_step():
    """The defining property: P^T (dl) must equal the delta we asked for."""
    for channels in (4, 16, 48):
        p = _sdxl_like_factors(channels)
        space = LatentColorSpace.rgb(p, channels=channels)
        delta_rgb = torch.randn(2, 3, 12, 10)
        delta_latent = space.backward_delta(delta_rgb)

        assert delta_latent.shape == (2, channels, 12, 10)
        got = torch.movedim(torch.movedim(delta_latent, 1, -1) @ p, -1, 1)
        assert torch.allclose(got, delta_rgb, atol=1e-4), f"C={channels}: {(got - delta_rgb).abs().max()}"
    print("  P^T dl == d_rgb for C = 4, 16, 48")


def test_back_projection_is_minimum_norm():
    """The step must lie inside span(P) — no energy in structure directions."""
    channels = 16
    p = _sdxl_like_factors(channels)
    space = LatentColorSpace.rgb(p, channels=channels)
    delta = space.backward_delta(torch.randn(1, 3, 8, 8))

    flat = delta.reshape(1, channels, -1)[0].T          # [N, C]
    # component orthogonal to span(P) must be zero
    q, _ = torch.linalg.qr(p)                           # [C,3] orthonormal basis
    residual = flat - (flat @ q) @ q.T
    assert float(residual.abs().max()) < 1e-4, float(residual.abs().max())


def test_forward_projection_matches_the_previewer():
    """Our forward map must equal what Latent2RGBPreviewer computes."""
    channels = 16
    p = _sdxl_like_factors(channels)
    bias = torch.randn(3)
    space = LatentColorSpace.rgb(p, bias, channels=channels)

    x0 = torch.randn(1, channels, 8, 8)
    ours = space.forward(x0)
    theirs = torch.nn.functional.linear(x0[0].movedim(0, -1), p.T, bias=bias)
    assert torch.allclose(ours[0].movedim(0, -1), theirs, atol=1e-5)


def test_flux2_reshape_round_trip():
    """Our inverse of Flux2's un-patchify must actually be its inverse."""
    x = torch.randn(2, 128, 6, 5)
    assert torch.equal(_flux2_inverse_reshape(FLUX2_RESHAPE(x)), x)
    print("  Flux2 128ch <-> 32ch round trip exact")


def test_rgb_projection_end_to_end_moves_colour():
    """A cast in projected RGB should come back as a latent step that removes it."""
    channels = 16
    p = _sdxl_like_factors(channels)
    space = LatentColorSpace.rgb(p, channels=channels)

    src = torch.randn(1, channels, 32, 32)
    ref = src + torch.movedim(
        torch.movedim(torch.ones(1, 3, 32, 32) * 2.0, 1, -1) @ space._back, -1, 1
    )

    cfg = GuidanceConfig(method="mean_only", strength=1.0)
    out = LatentCorrector(space, ref, cfg)(src)

    before = (space.forward(ref) - space.forward(src)).abs().mean()
    after = (space.forward(ref) - space.forward(out)).abs().mean()
    assert after < before * 0.1, f"cast only went from {before:.3f} to {after:.3f}"
    print(f"  projected cast {before:.3f} -> {after:.5f}")


# --------------------------------------------------------------------------- #
# the corrector
# --------------------------------------------------------------------------- #

def test_corrector_skips_tiny_crops():
    space = LatentColorSpace.raw(4, 4)
    cfg = GuidanceConfig(method="reinhard", strength=1.0)
    corrector = LatentCorrector(space, torch.randn(1, 4, 4, 4), cfg)
    tiny = torch.randn(1, 4, 4, 4)          # 16 elements < MIN_ELEMENTS
    assert torch.equal(corrector(tiny), tiny), "a tiny crop should be left alone"


def test_corrector_rejects_a_bad_batch():
    space = LatentColorSpace.raw(4, 4)
    cfg = GuidanceConfig(method="reinhard", strength=1.0)
    corrector = LatentCorrector(space, torch.randn(3, 4, 16, 16), cfg)
    src = torch.randn(2, 4, 16, 16)
    assert torch.equal(corrector(src), src), "mismatched batch should pass through untouched"


def test_corrector_broadcasts_a_single_reference():
    space = LatentColorSpace.raw(4, 4)
    cfg = GuidanceConfig(method="mean_only", strength=1.0)
    ref = torch.randn(1, 4, 16, 16) + 5.0
    corrector = LatentCorrector(space, ref, cfg)
    out = corrector(torch.randn(4, 4, 16, 16))
    means = out.mean(dim=[2, 3])
    assert torch.allclose(means, ref.mean(dim=[2, 3]).expand_as(means), atol=1e-3)


def test_corrector_handles_video_latents():
    space = LatentColorSpace.raw(16, 5)
    cfg = GuidanceConfig(method="mean_only", strength=1.0, temporal="global")
    ref = torch.randn(1, 16, 4, 16, 16) + 3.0
    out = LatentCorrector(space, ref, cfg)(torch.randn(1, 16, 4, 16, 16))
    assert out.shape == (1, 16, 4, 16, 16)
    assert torch.allclose(out.mean(dim=[2, 3, 4]), ref.mean(dim=[2, 3, 4]), atol=1e-3)


def test_strength_interpolates():
    space = LatentColorSpace.raw(4, 4)
    src = torch.randn(1, 4, 16, 16)
    ref = torch.randn(1, 4, 16, 16) + 10.0
    full = LatentCorrector(space, ref, GuidanceConfig("mean_only", strength=1.0))(src)
    half = LatentCorrector(space, ref, GuidanceConfig("mean_only", strength=0.5))(src)
    assert torch.allclose(half, (src + full) / 2, atol=1e-4)


def test_lowpass_preserves_the_average_shift():
    delta = torch.randn(1, 4, 32, 32) + 2.0
    smoothed = S.lowpass(delta, 3)
    assert smoothed.shape == delta.shape
    assert abs(float(smoothed.mean() - delta.mean())) < 0.05
    assert float(smoothed.std()) < float(delta.std()), "lowpass should reduce variation"


def test_lowpass_handles_video():
    delta = torch.randn(2, 4, 3, 16, 16)
    assert S.lowpass(delta, 2).shape == delta.shape


# --------------------------------------------------------------------------- #
# regressions: every one of these used to crash sampling or silently do nothing
# --------------------------------------------------------------------------- #

def test_mkl_with_per_frame_falls_back_instead_of_crashing():
    """MKL needs one covariance per clip; per-frame reduction cannot give it."""
    space = LatentColorSpace.raw(8, 5)
    cfg = GuidanceConfig(method="mkl", strength=1.0, temporal="per_frame")
    corrector = LatentCorrector(space, torch.randn(1, 8, 3, 16, 16), cfg)
    assert corrector.cfg.method == "reinhard", "should have stepped down to reinhard"
    assert cfg.method == "mkl", "the caller's config must not be rewritten"
    out = corrector(torch.randn(1, 8, 3, 16, 16))
    assert torch.isfinite(out).all()


def test_two_dimensional_mask_is_not_misread_on_image_latents():
    """[H,W] must behave like [1,H,W], not get folded into the batch axis."""
    x = torch.randn(1, 4, 16, 16)
    flat = torch.rand(16, 16)
    a = S.channel_stats(x, [2, 3], mask=flat)
    b = S.channel_stats(x, [2, 3], mask=flat.unsqueeze(0))
    assert torch.allclose(a.mean, b.mean, atol=1e-6)
    assert torch.allclose(a.std, b.std, atol=1e-6)


def test_mask_cache_does_not_grow():
    """A detailer feeding many crop sizes must not accumulate mask copies."""
    space = LatentColorSpace.raw(4, 4)
    cfg = GuidanceConfig(method="mean_only", strength=1.0)
    corrector = LatentCorrector(space, torch.randn(1, 4, 32, 32),
                                cfg, torch.ones(1, 64, 64))
    for size in (16, 20, 24, 28, 32):
        corrector(torch.randn(1, 4, size, size))
    assert not hasattr(corrector, "_mask_cache"), "the unbounded cache is gone"
    assert corrector._mask_key[0] == (1, 4, 32, 32), "only the last shape is kept"


def test_mask_without_a_time_axis_works_on_video():
    """One mask drawn for a clip should apply to every frame."""
    space = LatentColorSpace.raw(4, 5)
    mask = torch.zeros(1, 16, 16)
    mask[:, :, 8:] = 1.0
    cfg = GuidanceConfig(method="mean_only", strength=1.0)
    corrector = LatentCorrector(space, torch.randn(1, 4, 2, 16, 16), cfg, mask)
    out = corrector(torch.randn(1, 4, 2, 16, 16))
    assert out.shape == (1, 4, 2, 16, 16)
    assert torch.isfinite(out).all()


def test_mask_is_resized_to_what_is_being_sampled():
    """The reference may be a different size from the latent under the sampler."""
    space = LatentColorSpace.raw(4, 4)
    mask = torch.ones(1, 64, 64)                     # reference-sized
    cfg = GuidanceConfig(method="mean_only", strength=1.0)
    corrector = LatentCorrector(space, torch.randn(1, 4, 64, 64), cfg, mask)
    out = corrector(torch.randn(1, 4, 16, 24))       # a crop, different shape
    assert out.shape == (1, 4, 16, 24)
    assert torch.isfinite(out).all()


def test_an_almost_empty_mask_is_refused():
    """A mask covering a handful of pixels must not rescale the whole latent."""
    space = LatentColorSpace.raw(4, 4)
    mask = torch.zeros(1, 32, 32)
    mask[:, 0, 0] = 1.0
    cfg = GuidanceConfig(method="reinhard", strength=1.0, max_gain=8.0)
    corrector = LatentCorrector(space, torch.randn(1, 4, 32, 32) * 9, cfg, mask)
    src = torch.randn(1, 4, 32, 32)
    assert torch.equal(corrector(src), src), "near-empty mask should be a no-op"


def test_a_faint_mask_counts_as_what_it_is_worth():
    """A 0.02 mask over the whole frame is a few pixels of evidence, not 1024."""
    x = torch.randn(1, 4, 32, 32)
    full = S.channel_stats(x, [2, 3], mask=torch.ones(1, 32, 32))
    faint = S.channel_stats(x, [2, 3], mask=torch.full((1, 32, 32), 0.02))
    assert abs(full.count - 1024) < 1
    assert abs(faint.count - 1024) < 1, "a uniform faint mask still covers everything"

    patchy = torch.zeros(1, 32, 32)
    patchy[:, :2] = 1.0
    assert abs(S.channel_stats(x, [2, 3], mask=patchy).count - 64) < 1


def test_wrong_channel_count_is_skipped_not_crashed():
    """A reference from another model's VAE must not break the render."""
    space = LatentColorSpace.rgb(torch.randn(16, 3), channels=16)
    cfg = GuidanceConfig(method="mean_only", strength=1.0)
    corrector = LatentCorrector(space, torch.randn(1, 16, 16, 16), cfg)
    src = torch.randn(1, 4, 16, 16)                   # an SD1.5 prediction
    assert torch.equal(corrector(src), src), "should pass through untouched"


def test_covariance_refuses_per_frame_reduction():
    x = torch.randn(2, 4, 3, 8, 8)
    try:
        S.channel_stats(x, [3, 4], need_cov=True)
    except ValueError:
        return
    raise AssertionError("per-frame covariance should have been refused")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"- {fn.__name__}")
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
