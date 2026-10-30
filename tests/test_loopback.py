"""Tests for the Loopback node.

The sampler and the VAE are stubbed, so the loop's own logic — the fixed
reference, the denoise schedule, the seed walk, the feedback of each pass into
the next — is checked without a GPU or a checkpoint.

    python tests/test_loopback.py
"""

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# stand-ins for the parts that would need a model
# --------------------------------------------------------------------------- #

class FakeVAE:
    """Encodes by moving the channel axis and decodes by moving it back, so a
    pass through it is lossless and any colour change must come from sampling."""

    def encode(self, pixels):
        return pixels.movedim(-1, 1).clone()

    def decode(self, samples):
        return samples.movedim(1, -1).clone()


class FakeSampler:
    """Stands in for common_ksampler, applying a fixed multiplicative cast.

    Every call warms the image the same way, which is what an img2img chain
    does in miniature: without correction the cast compounds, with it the
    chain should hold still.
    """

    def __init__(self, cast=(1.10, 1.0, 0.92)):
        self.cast = cast
        self.calls = []

    def __call__(self, model, seed, steps, cfg, sampler_name, scheduler,
                 positive, negative, latent, denoise=1.0):
        self.calls.append({"seed": seed, "denoise": round(float(denoise), 6),
                           "has_mask": "noise_mask" in latent})
        samples = latent["samples"].clone()
        for channel, factor in enumerate(self.cast):
            samples[:, channel] = (samples[:, channel] * factor).clamp(0, 1)
        return ({"samples": samples},)


def install_stubs():
    """Put fake comfy modules on sys.path so the node imports without ComfyUI."""
    if "nodes" in sys.modules and hasattr(sys.modules["nodes"], "common_ksampler"):
        return sys.modules["nodes"]

    samplers = types.ModuleType("comfy.samplers")

    class KSampler:
        SAMPLERS = ["euler", "dpmpp_2m"]
        SCHEDULERS = ["normal", "karras"]

    samplers.KSampler = KSampler

    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update(self, n):
            pass

    utils.ProgressBar = ProgressBar

    comfy = types.ModuleType("comfy")
    comfy.samplers = samplers
    comfy.utils = utils

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.common_ksampler = None            # replaced per test

    sys.modules.setdefault("comfy", comfy)
    sys.modules["comfy.samplers"] = samplers
    sys.modules["comfy.utils"] = utils
    sys.modules["nodes"] = nodes_mod
    return nodes_mod


nodes_stub = install_stubs()


def import_loopback():
    """Import the node the way ComfyUI does — as part of the pack.

    ``loopback.py`` says ``from ..a1111_cc import color``, which only resolves
    inside a parent package, so one is built here rather than loosening the
    import in the node itself.
    """
    import importlib

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = types.ModuleType("a1111cc_pack")
    parent.__path__ = [root]
    sys.modules["a1111cc_pack"] = parent
    return importlib.import_module("a1111cc_pack.a1111_nodes.loopback")


_loopback = import_loopback()
A1111Loopback = _loopback.A1111Loopback
denoise_for_iteration = _loopback.denoise_for_iteration


def gradient_image(size=64):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    rgb = np.stack([0.30 + 0.4 * x, 0.45 + 0.3 * y, 0.60 - 0.3 * x * y], -1)
    return torch.from_numpy(np.clip(rgb, 0, 1).astype(np.float32)).unsqueeze(0)


def lab_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    from a1111_cc import color as C
    to_u8 = lambda t: np.rint(t[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    la = C.rgb_to_lab(to_u8(a)).reshape(-1, 3).astype(np.float32)
    lb = C.rgb_to_lab(to_u8(b)).reshape(-1, 3).astype(np.float32)
    return float(np.abs(la.mean(0) - lb.mean(0)).mean())


def run_loopback(loops=4, color_correction=True, cast=(1.10, 1.0, 0.92),
                 mask=None, seed=100, denoise=0.75, final_denoise=0.5,
                 curve="Linear", hold_exposure=False):
    sampler = FakeSampler(cast)
    nodes_stub.common_ksampler = sampler
    image = gradient_image()
    all_passes, final = A1111Loopback().run(
        model=None, vae=FakeVAE(), positive=None, negative=None, image=image,
        seed=seed, steps=4, cfg=7.0, sampler_name="euler", scheduler="normal",
        loops=loops, denoise=denoise, final_denoise=final_denoise,
        denoise_curve=curve, color_correction=color_correction,
        hold_exposure=hold_exposure, mask=mask,
    )
    return image, all_passes, final, sampler


# --------------------------------------------------------------------------- #
# the denoise schedule
# --------------------------------------------------------------------------- #

def test_denoise_schedule_matches_a1111():
    """Pinned against the arithmetic of A1111's own curve."""
    expected = {
        "Linear": [0.6667, 0.5833, 0.5],
        "Aggressive": [0.625, 0.5335, 0.5],
        "Lazy": [0.7165, 0.625, 0.5],
    }
    for curve, values in expected.items():
        got = [round(denoise_for_iteration(k, 4, 0.75, 0.5, curve), 4) for k in (1, 2, 3)]
        assert all(abs(a - b) < 1e-3 for a, b in zip(got, values)), f"{curve}: {got} != {values}"
    print("  Linear / Aggressive / Lazy all match")


def test_single_loop_keeps_the_initial_denoise():
    assert denoise_for_iteration(1, 1, 0.8, 0.2, "Linear") == 0.8


def test_last_pass_lands_exactly_on_final():
    for loops in (2, 3, 5, 9):
        last = denoise_for_iteration(loops - 1, loops, 0.9, 0.35, "Aggressive")
        assert abs(last - 0.35) < 1e-6, (loops, last)


def test_schedule_is_used_in_order():
    _, _, _, sampler = run_loopback(loops=4, curve="Linear")
    used = [c["denoise"] for c in sampler.calls]
    assert used[0] == 0.75, used
    assert abs(used[-1] - 0.5) < 1e-6, used
    assert used == sorted(used, reverse=True), "denoise should fall monotonically"
    print(f"  denoise per pass: {used}")


# --------------------------------------------------------------------------- #
# the point of the whole thing
# --------------------------------------------------------------------------- #

def test_correction_stops_the_cast_compounding():
    """Six passes of the same warm cast: uncorrected it stacks, corrected it does not."""
    image, _, drifted, _ = run_loopback(loops=6, color_correction=False)
    _, _, held, _ = run_loopback(loops=6, color_correction=True)

    drift = lab_distance(drifted, image)
    held_distance = lab_distance(held, image)
    assert held_distance < drift * 0.25, f"{held_distance:.2f} vs {drift:.2f}"
    print(f"  after 6 passes: uncorrected {drift:.2f}, corrected {held_distance:.2f}")


def _lab_split(a: torch.Tensor, b: torch.Tensor):
    """Distance between two images, separated into lightness and chroma."""
    from a1111_cc import color as C
    to_u8 = lambda t: np.rint(t[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    la = C.rgb_to_lab(to_u8(a)).reshape(-1, 3).astype(np.float32).mean(0)
    lb = C.rgb_to_lab(to_u8(b)).reshape(-1, 3).astype(np.float32).mean(0)
    diff = np.abs(la - lb)
    return float(diff[0]), float((diff[1] + diff[2]) / 2)


def test_chroma_is_held_flat_however_long_the_chain():
    """The reference never moves, so colour error must not grow with passes.

    Lightness is a different matter — see the test below.
    """
    image, _, short, _ = run_loopback(loops=2, color_correction=True)
    _, _, long_, _ = run_loopback(loops=10, color_correction=True)
    _, short_ab = _lab_split(short, image)
    _, long_ab = _lab_split(long_, image)

    assert long_ab < 1.0, f"chroma drifted to {long_ab:.2f} over 10 passes"
    assert long_ab < short_ab + 0.5, f"2 passes {short_ab:.2f}, 10 passes {long_ab:.2f}"
    print(f"  chroma error: 2 passes {short_ab:.2f}, 10 passes {long_ab:.2f} (flat)")


def test_a1111_correction_does_not_touch_exposure():
    """A property of A1111's blend, not a bug: it takes lightness from the
    generated image, so brightness drift passes through uncorrected."""
    image, _, held, _ = run_loopback(loops=6, color_correction=True)
    _, _, loose, _ = run_loopback(loops=6, color_correction=False)
    held_L, held_ab = _lab_split(held, image)
    loose_L, loose_ab = _lab_split(loose, image)

    assert abs(held_L - loose_L) < max(2.0, loose_L * 0.2), \
        f"lightness should pass through: {held_L:.2f} vs {loose_L:.2f}"
    assert held_ab < loose_ab * 0.1, f"chroma should be held: {held_ab:.2f} vs {loose_ab:.2f}"
    print(f"  lightness {held_L:.1f} vs {loose_L:.1f} (passes through), "
          f"chroma {held_ab:.2f} vs {loose_ab:.2f} (held)")


def test_hold_exposure_also_pins_lightness():
    """The optional extension: match L as well, which A1111 never did."""
    image, _, plain, _ = run_loopback(loops=6, color_correction=True)
    _, _, pinned, _ = run_loopback(loops=6, color_correction=True, hold_exposure=True)
    plain_L, _ = _lab_split(plain, image)
    pinned_L, pinned_ab = _lab_split(pinned, image)

    assert pinned_L < plain_L * 0.3, f"{pinned_L:.2f} vs {plain_L:.2f}"
    assert pinned_ab < 1.0, "chroma must stay held too"
    print(f"  lightness error {plain_L:.1f} -> {pinned_L:.2f} with hold_exposure")


def test_uncorrected_chain_really_does_drift():
    """Guard the test above: the stub must actually cause drift."""
    image, _, two, _ = run_loopback(loops=2, color_correction=False)
    _, _, eight, _ = run_loopback(loops=8, color_correction=False)
    assert lab_distance(eight, image) > lab_distance(two, image) * 1.5


# --------------------------------------------------------------------------- #
# the loop's mechanics
# --------------------------------------------------------------------------- #

def test_seed_advances_by_one_per_pass():
    _, _, _, sampler = run_loopback(loops=5, seed=1000)
    assert [c["seed"] for c in sampler.calls] == [1000, 1001, 1002, 1003, 1004]


def test_every_pass_is_returned_and_final_is_the_last():
    _, all_passes, final, _ = run_loopback(loops=4)
    assert all_passes.shape[0] == 4, all_passes.shape
    assert torch.equal(all_passes[-1:], final)


def test_single_pass_still_works():
    image, all_passes, final, sampler = run_loopback(loops=1)
    assert all_passes.shape[0] == 1
    assert len(sampler.calls) == 1
    assert sampler.calls[0]["denoise"] == 0.75


def test_mask_reaches_the_sampler():
    mask = torch.zeros(1, 64, 64)
    mask[:, :, 32:] = 1.0
    _, _, _, sampler = run_loopback(loops=3, mask=mask)
    assert all(c["has_mask"] for c in sampler.calls), "mask was dropped on some pass"


def test_output_stays_in_range_and_shape():
    image, all_passes, final, _ = run_loopback(loops=3)
    assert final.shape == image.shape
    assert float(final.min()) >= 0.0 and float(final.max()) <= 1.0
    assert final.dtype == image.dtype


def test_each_pass_feeds_the_next():
    """Pass N must sample from pass N-1, not from the original every time."""
    sampler = FakeSampler((1.05, 1.0, 0.97))
    nodes_stub.common_ksampler = sampler
    seen = []

    original_call = sampler.__call__

    def recording(model, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, latent, denoise=1.0):
        seen.append(latent["samples"].clone())
        return original_call(model, seed, steps, cfg, sampler_name, scheduler,
                             positive, negative, latent, denoise)

    nodes_stub.common_ksampler = recording
    A1111Loopback().run(
        model=None, vae=FakeVAE(), positive=None, negative=None,
        image=gradient_image(), seed=0, steps=4, cfg=7.0, sampler_name="euler",
        scheduler="normal", loops=3, denoise=0.75, final_denoise=0.5,
        denoise_curve="Linear", color_correction=False, hold_exposure=False,
        mask=None,
    )
    assert len(seen) == 3
    assert not torch.equal(seen[0], seen[1]), "second pass got the original image again"
    assert not torch.equal(seen[1], seen[2]), "third pass got a stale image"


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
