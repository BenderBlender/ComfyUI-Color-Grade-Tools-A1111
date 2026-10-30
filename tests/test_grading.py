"""Tests for the colour-grading core and the wheel mapping.

The interesting ones are not that the arithmetic runs, but that each control
does what a colourist expects it to: lift moves shadows and leaves white,
gain moves highlights and leaves black, gamma bends between them. If those
properties hold, the wheels behave like a grading panel rather than like
three sliders that all do roughly the same thing.

The wheel geometry is also checked against a port of the JavaScript, so the
widget and the backend cannot drift apart.

    python tests/test_grading.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a1111_cc import grading as G  # noqa: E402


def _node_module():
    """Import the node module as part of the pack.

    ``a1111_nodes/grading.py`` says ``from ..a1111_cc import glue``, which
    only resolves inside a parent package, so one is built here rather than
    loosening the import in the node itself.
    """
    import importlib
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "a1111cc_pack" not in sys.modules:
        parent = types.ModuleType("a1111cc_pack")
        parent.__path__ = [root]
        sys.modules["a1111cc_pack"] = parent
    return importlib.import_module("a1111cc_pack.a1111_nodes.grading")

SPAN = 0.5          # must match SPAN in web/colorwheels.js
TAU = math.pi * 2


# --------------------------------------------------------------------------- #
# the JavaScript, ported, so the two can be compared
# --------------------------------------------------------------------------- #

def js_rgb_from_handle(x, y):
    angle = math.atan2(y, x)
    radius = min(1.0, math.hypot(x, y))
    raw = [radius * math.cos(angle - (k * TAU) / 3) for k in range(3)]
    mean = sum(raw) / 3
    return [(v - mean) * SPAN for v in raw]


def js_handle_from_rgb(rgb):
    x = sum(rgb[k] * math.cos((k * TAU) / 3) for k in range(3)) * (2 / 3) / SPAN
    y = sum(rgb[k] * math.sin((k * TAU) / 3) for k in range(3)) * (2 / 3) / SPAN
    return x, y


def test_wheel_round_trip_is_exact():
    """A handle put on the disc must come back from the stored numbers."""
    worst = 0.0
    for angle in range(0, 360, 17):
        for radius in (0.15, 0.4, 0.75, 1.0):
            x = radius * math.cos(math.radians(angle))
            y = radius * math.sin(math.radians(angle))
            back = js_handle_from_rgb(js_rgb_from_handle(x, y))
            worst = max(worst, abs(back[0] - x), abs(back[1] - y))
    assert worst < 1e-9, worst
    print(f"  handle -> rgb -> handle, worst error {worst:.2e}")


def test_javascript_and_python_agree():
    """web/colorwheels.js and a1111_cc.grading must place colours identically."""
    worst = 0.0
    for angle in range(0, 360, 13):
        for radius in (0.25, 0.6, 1.0):
            x = radius * math.cos(math.radians(angle))
            y = radius * math.sin(math.radians(angle))
            from_js = js_rgb_from_handle(x, y)
            # Both measure the angle the same way; x and y here are the plain
            # mathematical pair, as the widget hands them over after dividing
            # out the disc radius.
            wheel = G.wheel_from_polar(angle, radius, gain=SPAN)
            from_py = [wheel.red, wheel.green, wheel.blue]
            worst = max(worst, max(abs(a - b) for a, b in zip(from_js, from_py)))
    assert worst < 1e-9, worst
    print(f"  JS and Python agree to {worst:.2e}")


def test_a_wheel_is_purely_chromatic():
    """Moving a handle must not change overall brightness — that is the ring's job."""
    for angle in (0, 45, 130, 250, 300):
        rgb = js_rgb_from_handle(math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        assert abs(sum(rgb)) < 1e-9, (angle, sum(rgb))


def test_red_handle_pushes_red():
    rgb = js_rgb_from_handle(1.0, 0.0)
    assert rgb[0] > 0 and rgb[1] < 0 and rgb[2] < 0, rgb


# --------------------------------------------------------------------------- #
# what each control is for
# --------------------------------------------------------------------------- #

def ramp():
    """A black-to-white ramp, so each control can be watched across the range."""
    values = torch.linspace(0.0, 1.0, 256).reshape(1, 1, 1, 256)
    return values.repeat(1, 3, 1, 1)


def test_lift_moves_shadows_and_leaves_white():
    graded = G.apply_grade(ramp(), G.Grade(lift=G.Wheel(master=0.2)))
    base = ramp()
    shadow = float((graded[..., 8] - base[..., 8]).mean())
    white = float((graded[..., -1] - base[..., -1]).mean())
    assert shadow > 0.15, shadow
    assert abs(white) < 1e-5, white
    print(f"  lift: shadows +{shadow:.3f}, white {white:+.5f}")


def test_gain_moves_highlights_and_leaves_black():
    graded = G.apply_grade(ramp(), G.Grade(gain=G.Wheel(master=0.25)))
    base = ramp()
    black = float((graded[..., 0] - base[..., 0]).mean())
    white = float((graded[..., -1] - base[..., -1]).mean())
    assert white > 0.2, white
    assert abs(black) < 1e-5, black
    print(f"  gain: white +{white:.3f}, black {black:+.5f}")


def test_gamma_bends_midtones_most():
    graded = G.apply_grade(ramp(), G.Grade(gamma=G.Wheel(master=0.4)))
    base = ramp()
    delta = (graded - base)[0, 0, 0]
    mid = float(delta[128].abs())
    ends = float(max(delta[2].abs(), delta[-2].abs()))
    assert mid > ends * 4, f"mid {mid:.4f} against ends {ends:.4f}"
    print(f"  gamma: midtone shift {mid:.3f}, ends {ends:.4f}")


def test_saturation_zero_is_monochrome():
    rgb = torch.rand(1, 3, 8, 8)
    graded = G.apply_grade(rgb, G.Grade(saturation=0.0))
    spread = float((graded.max(dim=1).values - graded.min(dim=1).values).max())
    assert spread < 1e-5, spread


def test_contrast_pivots_on_mid_grey():
    grey = torch.full((1, 3, 4, 4), 0.5)
    graded = G.apply_grade(grey, G.Grade(contrast=2.0))
    assert torch.allclose(graded, grey, atol=1e-6)


def test_neutral_grade_changes_nothing():
    rgb = torch.rand(1, 3, 16, 16)
    assert G.Grade().is_neutral()
    assert torch.allclose(G.apply_grade(rgb, G.Grade()), rgb, atol=1e-6)


def test_unit_range_round_trip():
    x = torch.linspace(-1.0, 1.0, 64).reshape(1, 1, 1, 64).repeat(1, 3, 1, 1)
    assert torch.allclose(G.from_unit(G.to_unit(x)), x, atol=1e-6)
    assert torch.allclose(G.grade_unit_range(x, G.Grade()), x, atol=1e-6)


def test_gamma_never_divides_by_zero():
    graded = G.apply_grade(ramp(), G.Grade(gamma=G.Wheel(master=-5.0)))
    assert torch.isfinite(graded).all()


def test_video_shaped_tensors_are_handled():
    rgb = torch.rand(1, 3, 4, 8, 8)
    graded = G.apply_grade(rgb, G.Grade(lift=G.Wheel(red=0.1), saturation=1.3))
    assert graded.shape == rgb.shape
    assert torch.isfinite(graded).all()


def test_lift_and_gain_are_independent_endpoints():
    """Black lands on lift and white on gain, whatever the other one is doing.

    This is the property that makes the two wheels feel like separate
    controls. Composing them as `v + lift*(1-v)` and then `v*gain` instead
    would let gain drag the black point around whenever lift was non-zero.
    """
    base = ramp()
    lift, gain = 0.15, 1.2

    graded = G.apply_grade(base, G.Grade(lift=G.Wheel(master=lift),
                                         gain=G.Wheel(master=gain - 1.0)))
    black = float(graded[..., 0].mean())
    white = float(graded[..., -1].mean())
    assert abs(black - lift) < 1e-5, black
    assert abs(white - gain) < 1e-5, white
    print(f"  black -> {black:.3f} (lift), white -> {white:.3f} (gain)")


def test_gamma_passes_negatives_through():
    """A power curve has nothing to say below zero, and clamping there would
    crush the darkest part of the latent a little more on every step."""
    values = torch.tensor([-0.3, -0.1, 0.0, 0.25, 0.5]).reshape(1, 1, 1, 5).repeat(1, 3, 1, 1)
    graded = G.apply_grade(values, G.Grade(gamma=G.Wheel(master=0.5)))
    assert torch.allclose(graded[..., :2], values[..., :2], atol=1e-6), graded[0, 0, 0]
    assert float(graded[..., 3].mean()) != float(values[..., 3].mean())
    print(f"  negatives untouched: {[round(float(v), 3) for v in graded[0, 0, 0]]}")


def test_gamma_continues_smoothly_above_white():
    """Above one the curve follows its own tangent instead of exploding."""
    values = torch.tensor([1.0, 1.5, 2.0]).reshape(1, 1, 1, 3).repeat(1, 3, 1, 1)
    graded = G.apply_grade(values, G.Grade(gamma=G.Wheel(master=-0.5)))[0, 0, 0]
    assert abs(float(graded[0]) - 1.0) < 1e-6
    steps = [float(graded[1] - graded[0]), float(graded[2] - graded[1])]
    assert abs(steps[0] - steps[1]) < 1e-5, steps      # linear, so even steps
    assert float(graded[2]) < 4.0, float(graded[2])    # a power curve would reach 4


def test_defaults_are_neutral():
    """An untouched grading node must change nothing at all.

    It did not: the node added the neutral centre on top of a value that had
    already had it taken off, so a fresh node graded at gamma 2 and gain 2.
    """
    N = _node_module()

    defaults = {name: spec[1]["default"] for name, spec in N.GRADE_CONTROLS.items()}
    grade = N._grade_from(defaults)

    assert grade.is_neutral(), (grade.gamma.master, grade.gain.master, grade.lift.master)
    base = ramp()
    assert torch.allclose(G.apply_grade(base, grade), base, atol=1e-6), \
        [round(float(v), 3) for v in G.apply_grade(base, grade)[0, 0, 0, ::64]]
    print("  a fresh node is a no-op")


def test_widget_values_reach_the_grade():
    """A master typed into the node must arrive as an offset from neutral."""
    N = _node_module()

    values = {name: spec[1]["default"] for name, spec in N.GRADE_CONTROLS.items()}
    values["gain_master"] = 1.25
    values["lift_master"] = 0.1
    values["gamma_master"] = 0.8
    grade = N._grade_from(values)

    assert abs(grade.gain.master - 0.25) < 1e-9, grade.gain.master
    assert abs(grade.lift.master - 0.10) < 1e-9, grade.lift.master
    assert abs(grade.gamma.master + 0.20) < 1e-9, grade.gamma.master

    # The tensors come back in the source dtype, so compare at float32's
    # resolution rather than the double-precision one used above.
    probe = torch.zeros(1, 3, 1, 1)
    assert abs(float(grade.gain.as_tensor(probe, centre=1.0)[0, 0]) - 1.25) < 1e-6
    assert abs(float(grade.gamma.as_tensor(probe, centre=1.0)[0, 0]) - 0.80) < 1e-6
    print("  typed values arrive as offsets from neutral")


# --------------------------------------------------------------------------- #
# grading towards levels
# --------------------------------------------------------------------------- #

def _space(channels=4):
    from a1111_cc import space as S
    torch.manual_seed(7)
    return S.LatentColorSpace.rgb(torch.randn(channels, 3) * 0.3, channels=channels)


def test_levels_land_on_the_reference():
    """With an anchor the image takes that picture's shadows and highlights."""
    source = torch.rand(1, 3, 48, 48) * 0.4 + 0.1
    reference = torch.rand(1, 3, 32, 32) * 0.9 + 0.05
    out = G.grade_levels(source, G.Grade(), reference)

    got = G.channel_levels(out)
    want = G.channel_levels(reference)
    assert torch.allclose(got, want, atol=0.03), (got, want)
    print(f"  mean {float(source.mean()):.3f} -> {float(out.mean()):.3f} "
          f"(reference {float(reference.mean()):.3f})")


def test_matching_is_idempotent():
    """Applied twice it must say the same thing — that is what converging means."""
    source = torch.rand(1, 3, 48, 48)
    reference = torch.rand(1, 3, 48, 48) * 0.6 + 0.2
    once = G.grade_levels(source, G.Grade(), reference)
    twice = G.grade_levels(once, G.Grade(), reference)
    assert torch.allclose(once, twice, atol=1e-3), float((once - twice).abs().max())


def test_a_flat_source_is_shifted_not_stretched():
    """Dividing by a tiny range would multiply contrast by a hundred."""
    flat = torch.full((1, 3, 16, 16), 0.5)
    reference = torch.rand(1, 3, 16, 16)
    out = G.grade_levels(flat, G.Grade(), reference)
    assert torch.isfinite(out).all()
    assert float(out.std()) < 0.1, float(out.std())
    print(f"  flat source keeps its shape: std {float(out.std()):.4f}")


def test_slopes_are_clamped():
    """A tight source must not be stretched without limit onto a wide target."""
    narrow = torch.rand(1, 3, 32, 32) * 0.08 + 0.46       # a very tight range
    wide = torch.rand(1, 3, 32, 32)
    out = G.grade_levels(narrow, G.Grade(), wide)

    before = G.channel_levels(narrow)
    after = G.channel_levels(out)
    stretch = float(((after[..., 2] - after[..., 0]) /
                     (before[..., 2] - before[..., 0])).max())
    assert stretch <= G.MAX_SLOPE + 0.01, stretch
    print(f"  range stretched {stretch:.2f}x, capped at {G.MAX_SLOPE}")


def test_the_map_has_no_step_in_it():
    """Clamping the slopes must not leave the two halves disagreeing.

    Anchoring each half at its own outer knot looked right and was not: with
    the slopes capped the halves arrived at different values either side of
    the midtone, a jump of 0.29 straight through the midtones.
    """
    source = torch.tensor([[[0.46, 0.50, 0.54]]])
    target = torch.tensor([[[0.05, 0.49, 0.95]]])          # needs slope 12

    probe = torch.linspace(0.3, 0.7, 4001).reshape(1, 1, 1, -1)
    mapped = G.piecewise_map(probe, source, target).flatten()
    jumps = (mapped[1:] - mapped[:-1]).abs()
    assert float(jumps.max()) < 0.01, float(jumps.max())

    # and it still passes exactly through the midtone it was asked for
    mid = G.piecewise_map(source[..., 1].reshape(1, 1, 1, 1), source, target)
    assert abs(float(mid) - 0.49) < 1e-5, float(mid)
    print(f"  largest step across the curve: {float(jumps.max()):.2e}")


def test_an_unclamped_map_hits_every_knot():
    """When nothing is capped the curve passes through all three targets."""
    source = torch.tensor([[[0.20, 0.50, 0.80]]])
    target = torch.tensor([[[0.30, 0.55, 0.75]]])          # gentle, no clamping
    probe = source.reshape(1, 1, 1, 3)
    got = G.piecewise_map(probe, source, target).flatten()
    assert torch.allclose(got, target.flatten(), atol=1e-5), got


def test_saturation_is_not_run_through_percentiles():
    """It mixes channels, so three per-channel levels cannot express it."""
    rgb = torch.rand(1, 3, 32, 32)
    out = G.grade_levels(rgb, G.Grade(saturation=0.0), rgb)
    spread = float((out.max(dim=1).values - out.min(dim=1).values).max())
    assert spread < 1e-4, spread


def test_a_single_reference_serves_a_batch():
    out = G.grade_levels(torch.rand(4, 3, 16, 16), G.Grade(), torch.rand(1, 3, 16, 16))
    assert out.shape == (4, 3, 16, 16)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# the sampling hook
# --------------------------------------------------------------------------- #

def test_a_neutral_grade_survives_a_changing_prediction():
    """The reason the first attempt at this was wrong.

    Early in sampling x0 is blurred and its spread is a fraction of the
    finished image's. Anchoring to it would squeeze contrast a little further
    on every step — so a neutral grade is checked against exactly that.
    """
    N = _node_module()
    corrector = N.GradeCorrector(_space(), G.Grade(), 0.5)

    worst = 0.0
    for scale in (0.3, 0.3, 1.0, 1.0, 1.0, 1.0):
        x = torch.randn(1, 4, 32, 32) * scale
        out = corrector(x, None)
        worst = max(worst, float((out - x).abs().max()),
                    abs(float(out.std() - x.std())))
    assert worst < 1e-4, worst
    print(f"  neutral grade moves nothing: worst {worst:.2e}")


def test_the_hook_converges_instead_of_compounding():
    N = _node_module()
    grade = G.Grade(gain=G.Wheel(red=0.10), lift=G.Wheel(master=0.05))
    corrector = N.GradeCorrector(_space(), grade, 0.5)

    x = torch.randn(1, 4, 32, 32)
    steps = []
    for _ in range(12):
        nxt = corrector(x, None)
        steps.append(float((nxt - x).abs().mean()))
        x = nxt

    assert steps[-1] < steps[0] * 0.02, steps
    assert all(b <= a + 1e-6 for a, b in zip(steps, steps[1:])), steps
    print(f"  steps {steps[0]:.3f} -> {steps[3]:.3f} -> {steps[-1]:.5f}")


def test_state_is_dropped_when_a_new_run_starts():
    """ComfyUI hands back the same patched model while the inputs are
    unchanged, so one seed's targets must not steer the next."""
    N = _node_module()
    corrector = N.GradeCorrector(_space(), G.Grade(gain=G.Wheel(red=0.1)), 0.5)

    x = torch.randn(1, 4, 32, 32)
    for sigma in (14.0, 7.0, 3.0, 1.0):
        x = corrector(x, sigma)
    assert corrector._injected is not None

    corrector(torch.randn(1, 4, 32, 32), 14.0)      # sigma jumped back up
    assert corrector._last_sigma == 14.0


def test_a_mid_run_sigma_wobble_does_not_reset():
    """Samplers that evaluate the model twice per step report the same sigma
    twice, and some move it slightly; only a real jump counts."""
    N = _node_module()
    corrector = N.GradeCorrector(_space(), G.Grade(gain=G.Wheel(red=0.1)), 0.5)
    x = torch.randn(1, 4, 32, 32)
    for sigma in (12.0, 12.0, 11.8, 11.9, 9.0, 9.0):
        x = corrector(x, sigma)
        assert corrector._injected is not None, sigma


def test_an_anchor_fixes_the_targets():
    N = _node_module()
    space = _space()
    anchor = space.forward(torch.randn(1, 4, 32, 32) * 0.7)
    corrector = N.GradeCorrector(space, G.Grade(), 1.0, anchor)

    first = corrector(torch.randn(1, 4, 32, 32), 10.0)
    second = corrector(torch.randn(1, 4, 32, 32), 5.0)
    assert corrector._injected is None, "an anchored grade needs no injection tracking"
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_anchor_and_wheels_together():
    N = _node_module()
    space = _space()
    anchor = space.forward(torch.randn(1, 4, 32, 32))
    plain = N.GradeCorrector(space, G.Grade(), 1.0, anchor)
    warmed = N.GradeCorrector(space, G.Grade(gain=G.Wheel(red=0.12)), 1.0, anchor)

    x = torch.randn(1, 4, 32, 32)
    a = space.forward(plain(x, 5.0))
    b = space.forward(warmed(x, 5.0))
    assert float(b[:, 0].mean()) > float(a[:, 0].mean()), "the wheel should sit on top"


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
