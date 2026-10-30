"""Unit tests for the image-space colour maths.

Runs under pytest, or directly::

    python tests/test_color.py

scikit-image is only needed for the parity test; it is skipped when missing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a1111_cc import color as C  # noqa: E402

RNG = np.random.default_rng(1234)


def _random_pair(h=96, w=128):
    src = RNG.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    ref = np.clip(RNG.normal(150, 40, size=(h, w, 3)), 0, 255).astype(np.uint8)
    return src, ref


# --------------------------------------------------------------------------- #
# histogram matching
# --------------------------------------------------------------------------- #

def test_match_histograms_matches_scikit_image():
    """Our LUT must equal scikit-image's result exactly. This test used to allow
    one level — the very «deliberate difference» (rounding where scikit-image
    truncates) that kept the node off the webui; a check must not borrow the
    rule of the code it checks (30.10)."""
    try:
        from skimage.exposure import match_histograms
    except ImportError:
        print("  (skipped: scikit-image not installed)")
        return

    worst = 0
    for _ in range(5):
        src, ref = _random_pair()
        mine = C.match_histograms_u8(src, ref)
        theirs = match_histograms(src, ref, channel_axis=2).astype(np.uint8)
        worst = max(worst, int(np.abs(mine.astype(int) - theirs.astype(int)).max()))
    assert worst == 0, f"diverged from scikit-image by {worst} levels"
    print(f"  identical to scikit-image on 5 random pairs")


def test_luminosity_blend_matches_blendmodes():
    """The blend must equal blendmodes.blendLayers(corrected, original,
    LUMINOSITY) exactly, as the webui calls it. Runs where blendmodes is
    installed (DEMO_RESULTS/venv_a1111ref has the webui's pinned 2022)."""
    try:
        from blendmodes.blend import blendLayers, BlendType
        from PIL import Image
    except ImportError:
        print("  (skipped: blendmodes not installed)")
        return
    rng = np.random.default_rng(7)
    worst = 0
    for _ in range(3):
        a = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        theirs = np.asarray(blendLayers(Image.fromarray(a), Image.fromarray(b), BlendType.LUMINOSITY).convert("RGB"))
        mine = C.a1111_luminosity_blend(a, b)
        worst = max(worst, int(np.abs(mine.astype(int) - theirs.astype(int)).max()))
    assert worst == 0, f"diverged from blendmodes by {worst} levels"
    print("  identical to blendmodes on 3 random pairs")


def test_match_histograms_is_idempotent_against_itself():
    """Matching an image to itself must be a no-op."""
    src, _ = _random_pair()
    out = C.match_histograms_u8(src, src)
    assert np.array_equal(out, src), "self-match changed the image"


def test_match_histograms_transfers_the_distribution():
    """After matching, the per-channel mean should land near the reference's."""
    src, ref = _random_pair(256, 256)
    out = C.match_histograms_u8(src, ref)
    for c in range(3):
        assert abs(float(out[..., c].mean()) - float(ref[..., c].mean())) < 2.0


def test_masked_match_uses_only_selected_pixels():
    """Statistics must come from the mask, even when the rest of the frame differs."""
    h = w = 128
    src = np.full((h, w, 3), 100, dtype=np.uint8)
    src[:, : w // 2] = 10  # a dark half that must not influence the match
    ref = np.full((h, w, 3), 200, dtype=np.uint8)

    sel = np.zeros((h, w), dtype=bool)
    sel[:, w // 2:] = True  # measure on the uniform 100-valued half only

    out = C.match_histograms_masked_u8(src, ref, src_mask=sel)
    assert out[0, w // 2, 0] == 200, "masked region was not mapped onto the reference"

    # Levels that never appear inside the mask still get a mapping, taken from
    # the nearest level below. Leaving them as themselves would be a step in
    # the middle of the curve, visible wherever a soft mask edge spans it.
    # Outside the mask it does not matter: the node blends by mask weight.
    lut_is_monotonic = np.all(np.diff(
        C.match_histograms_masked_u8(
            np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, 2),
            ref, src_mask=np.ones((1, 256), dtype=bool))[0, :, 0].astype(int)
    ) >= 0)
    assert lut_is_monotonic, "the mapping must never go backwards"


# --------------------------------------------------------------------------- #
# the A1111 blend
# --------------------------------------------------------------------------- #

def test_a1111_blend_keeps_the_originals_brightness():
    """The defining property: luma comes from the ORIGINAL, not the corrected image.

    This is the behaviour the first version of the node had backwards.
    """
    corrected, original = _random_pair()
    out = C.a1111_luminosity_blend(corrected, original)

    luma = lambda u8: (u8.astype(np.float32) / 255.0) @ np.array([0.3, 0.59, 0.11], np.float32)
    err = np.abs(luma(out) - luma(original))
    # out-of-gamut pixels get pulled back by _clipColor, so allow a small tail
    assert float(np.median(err)) < 0.01, f"median luma error {float(np.median(err)):.4f}"
    assert float(err.mean()) < 0.05, f"mean luma error {float(err.mean()):.4f}"
    print(f"  luma error vs original: median={np.median(err):.5f} mean={err.mean():.5f}")


def test_a1111_blend_takes_chroma_from_the_corrected_image():
    """A grey original plus a coloured correction must come out coloured."""
    grey = np.full((32, 32, 3), 128, dtype=np.uint8)
    warm = np.dstack([
        np.full((32, 32), 200, np.uint8),
        np.full((32, 32), 120, np.uint8),
        np.full((32, 32), 60, np.uint8),
    ]).astype(np.uint8)

    out = C.a1111_luminosity_blend(warm, grey)
    spread = int(out[..., 0].mean()) - int(out[..., 2].mean())
    assert spread > 50, f"chroma was not carried over (R-B spread {spread})"


def test_a1111_blend_is_the_opposite_of_lab_luminosity():
    """Guard against silently reintroducing the inverted blend."""
    corrected, original = _random_pair()
    a1111 = C.a1111_luminosity_blend(corrected, original)
    inverted = C.lab_luminosity_blend(corrected, original)
    diff = np.abs(a1111.astype(int) - inverted.astype(int)).mean()
    assert diff > 5.0, f"the two blends have collapsed onto each other (mean diff {diff:.2f})"
    print(f"  a1111 vs inverted blend: mean difference {diff:.1f}/255")


def test_a1111_blend_is_stable_at_the_extremes():
    """Pure black and pure white must not produce NaNs through _clipColor."""
    for value in (0, 255):
        flat = np.full((16, 16, 3), value, dtype=np.uint8)
        other = RNG.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        assert np.isfinite(C.a1111_luminosity_blend(flat, other).astype(float)).all()
        assert np.isfinite(C.a1111_luminosity_blend(other, flat).astype(float)).all()


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def test_lab_round_trip_is_close():
    src, _ = _random_pair()
    back = C.lab_to_rgb(C.rgb_to_lab(src))
    assert np.abs(back.astype(int) - src.astype(int)).mean() < 2.0


def test_even_blur_radius_is_not_ignored():
    """An even radius used to silently do nothing."""
    src, _ = _random_pair()
    assert not np.array_equal(C.gaussian_blur_u8(src, 4), src)
    assert np.array_equal(C.gaussian_blur_u8(src, 0), src)


def test_numpy_lab_matches_opencv():
    """The dependency-free LAB path must agree with OpenCV's.

    OpenCV is preferred when installed, because Automatic1111 runs on it, but
    the pack must install with nothing beyond numpy — so the fallback has to
    land in the same place.
    """
    if C.cv2 is None:
        print("  (skipped: OpenCV not installed, the fallback is the only path)")
        return

    # A colour grid plus the dark corner, where OpenCV's 12-bit lookup tables
    # and a closed-form conversion disagree most.
    grid = np.stack(np.meshgrid(*[np.arange(0, 256, 5)] * 3, indexing="ij"), -1)
    dark = np.stack(np.meshgrid(*[np.arange(0, 25)] * 3, indexing="ij"), -1)
    cases = [np.ascontiguousarray(g.reshape(1, -1, 3).astype(np.uint8))
             for g in (grid, dark)]
    cases.append(_random_pair(128, 128)[0])

    worst = 0
    for src in cases:
        ours = C._rgb_to_lab_numpy(src)
        theirs = C.cv2.cvtColor(src, C.cv2.COLOR_RGB2LAB)
        worst = max(worst, int(np.abs(ours.astype(int) - theirs.astype(int)).max()))
    assert worst <= 3, f"LAB diverged by {worst} levels"
    diff = np.array([worst])

    back_ours = C._lab_to_rgb_numpy(theirs)
    back_theirs = C.cv2.cvtColor(theirs, C.cv2.COLOR_LAB2RGB)
    back_diff = np.abs(back_ours.astype(int) - back_theirs.astype(int))
    assert back_diff.max() <= 3, f"LAB->RGB diverged by {back_diff.max()} levels"
    print(f"  numpy vs OpenCV: RGB->LAB max {diff.max()}, LAB->RGB max {back_diff.max()}")


def test_numpy_lab_round_trip():
    src, _ = _random_pair()
    back = C._lab_to_rgb_numpy(C._rgb_to_lab_numpy(src))
    assert np.abs(back.astype(int) - src.astype(int)).mean() < 2.0


def test_resize_fallback_matches_opencv():
    if C.cv2 is None:
        return
    arr = RNG.random((37, 53)).astype(np.float32)
    cv_out = C.cv2.resize(arr, (24, 16), interpolation=C.cv2.INTER_LINEAR)
    saved, C.cv2 = C.cv2, None
    try:
        np_out = C.resize_bilinear(arr, 24, 16)
    finally:
        C.cv2 = saved
    assert np.abs(np_out - cv_out).max() < 0.05, np.abs(np_out - cv_out).max()


def test_mix_endpoints():
    a, b = _random_pair()
    assert np.array_equal(C.mix_u8(a, b, 0.0), a)
    assert np.array_equal(C.mix_u8(a, b, 1.0), b)


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
