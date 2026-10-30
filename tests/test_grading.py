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

def test_a_neutral_grade_never_reaches_the_sampler():
    """Nothing is hooked at all when every wheel is at rest, so the question
    of what a neutral grade would do to a prediction does not arise."""
    N = _node_module()
    defaults = {name: spec[1]["default"] for name, spec in N.GRADE_CONTROLS.items()}
    assert N._grade_from(defaults).is_neutral()


def test_a_fixed_target_holds_the_levels_it_was_given():
    """Holding on to levels is the meaning of the control, not a defect.

    Once a target is set the hook keeps pulling towards it, so a prediction
    that drifts afterwards is pulled back. That is what "put the shadows
    here" asks for. It is also why the window starts a third of the way in:
    a target taken from the blur at the top of the schedule would be a target
    for the wrong picture, and everything after it would be dragged towards
    that blur.
    """
    N = _node_module()
    space = _space()
    grade = G.Grade(gain=G.Wheel(red=0.10))
    corrector = N.GradeCorrector(space, grade, 1.0)

    settled = torch.randn(1, 4, 32, 32)
    corrector(settled, 4.0)
    targets = corrector._targets.clone()

    out = torch.randn(1, 4, 32, 32) * 1.4          # drifted; each call closes RATE of the gap
    for sigma in (3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6):
        out = corrector(out, sigma)

    # measured the way the hook measures: levels of the blurred projection
    reached = G.channel_levels(corrector._smooth(G.to_unit(space.forward(out))))
    assert torch.allclose(reached, targets, atol=0.05), float((reached - targets).abs().max())
    assert torch.allclose(corrector._targets, targets), "the target must not move"
    print(f"  later drift pulled back to the target, within "
          f"{float((reached - targets).abs().max()):.3f}")


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


def test_saturation_converges_instead_of_compounding():
    """The test above, with the one control it never exercised.

    Saturation was applied as a factor on every call, so it compounded while
    the levels converged. On SDXL a 1.15 latent grade raised the finished
    image's colour 3.2-fold and turned a dusk street neon; the same grade
    applied afterwards raises it 1.4-fold. A model that keeps everything it is
    given is the worst case: here the colour must settle near 1.15 times the
    first prediction's, and each call must ask for less than the one before.
    """
    N = _node_module()
    space = _space()
    corrector = N.GradeCorrector(space, G.Grade(saturation=1.15), 1.0)

    def colour(latent):
        # measured here, independently of the code under test: mean |rgb - grey|
        rgb = G.to_unit(space.forward(latent))
        grey = (rgb * torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        return float((rgb - grey).abs().mean())

    x = torch.randn(1, 4, 32, 32)
    start = colour(x)
    steps = []
    for i in range(17):
        nxt = corrector(x, 10.0 - i * 0.5)
        steps.append(float((nxt - x).abs().mean()))
        x = nxt
    end = colour(x)

    ratio = end / start
    assert 1.05 < ratio < 1.25, f"colour went x{ratio:.2f}; 1.15 was asked for"
    assert steps[-1] < steps[0] * 0.05, steps
    print(f"  colour x{ratio:.3f} after 17 calls (asked 1.15); "
          f"step {steps[0]:.4f} -> {steps[-1]:.6f}")


def test_neutral_wheels_with_an_anchor_still_patch_the_model():
    """An anchor aims the grade at its levels, so it acts even with every wheel
    at neutral. The node returned the model untouched in that case — the anchor
    did nothing at all — while ImageColorGrade, testing its reference the same
    way, always worked."""
    N = _node_module()
    controls = {k: v for k, v in N.LatentColorGrade.INPUT_TYPES()["required"].items()}
    neutral = {name: spec[1]["default"] for name, spec in controls.items()
               if name not in ("model", "strength", "start_percent", "end_percent")}

    class ReachedThePatch(Exception):
        pass

    class Model:
        def get_model_object(self, name):
            raise ReachedThePatch(name)

    original = N.glue.detach
    N.glue.detach = lambda model, key=None: "left untouched"
    try:
        node = N.LatentColorGrade()
        unanchored = node.apply(Model(), 0.5, 0.3, 1.0, **neutral)[0]
        assert unanchored == "left untouched", "neutral and unanchored must be a no-op"
        try:
            node.apply(Model(), 0.5, 0.3, 1.0, anchor={"samples": torch.zeros(1, 4, 8, 8)},
                       **neutral)
        except ReachedThePatch:
            pass
        else:
            raise AssertionError("neutral wheels with an anchor left the model untouched")
    finally:
        N.glue.detach = original


def test_state_is_dropped_when_a_new_run_starts():
    """ComfyUI hands back the same patched model while the inputs are
    unchanged, so one seed's targets must not steer the next."""
    N = _node_module()
    corrector = N.GradeCorrector(_space(), G.Grade(gain=G.Wheel(red=0.1)), 0.5)

    x = torch.randn(1, 4, 32, 32)
    for sigma in (14.0, 7.0, 3.0, 1.0):
        x = corrector(x, sigma)
    first_targets = corrector._targets.clone()

    corrector(torch.randn(1, 4, 32, 32), 14.0)      # sigma jumped back up
    assert corrector._last_sigma == 14.0
    assert not torch.allclose(corrector._targets, first_targets),         "the second run must take its own bearings"


def test_a_mid_run_sigma_wobble_does_not_reset():
    """Samplers that evaluate the model twice per step report the same sigma
    twice, and some move it slightly; only a real jump counts."""
    N = _node_module()
    corrector = N.GradeCorrector(_space(), G.Grade(gain=G.Wheel(red=0.1)), 0.5)
    x = torch.randn(1, 4, 32, 32)
    corrector(x, 12.0)
    targets = corrector._targets.clone()
    for sigma in (12.0, 11.8, 11.9, 9.0, 9.0):
        x = corrector(x, sigma)
        assert torch.allclose(corrector._targets, targets), sigma


def test_an_anchor_fixes_the_targets():
    N = _node_module()
    space = _space()
    anchor = space.forward(torch.randn(1, 4, 32, 32) * 0.7)
    corrector = N.GradeCorrector(space, G.Grade(), 1.0, anchor)

    first = corrector(torch.randn(1, 4, 32, 32), 10.0)
    targets = corrector._targets.clone()
    second = corrector(torch.randn(1, 4, 32, 32), 5.0)
    assert torch.allclose(corrector._targets, targets),         "an anchored grade must not re-aim at each prediction"
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


def test_the_grade_survives_a_model_that_pushes_back():
    """The property the node exists for, and the one that was missing.

    A sampler does not simply accept what the hook hands it: the model
    re-predicts, and some of the correction is undone. Holding a fixed target
    means every call pulls towards the same place, so what arrives is roughly
    proportional to how much the model cooperates. Estimating the ungraded
    image instead — by subtracting back what had been injected — looked
    equivalent and was not: it reads its own correction as the model's work,
    stops pushing, and delivers almost nothing.
    """
    N = _node_module()
    space = _space()
    grade = G.Grade(gain=G.Wheel(red=0.12), lift=G.Wheel(master=0.05))

    def delivered(keeps, calls=16):
        corrector = N.GradeCorrector(space, grade, 1.0)     # the whole grade asked for
        base = torch.randn(1, 4, 32, 32)
        x = base.clone()
        for i in range(calls):
            injected = corrector(x, 10.0 - i * 0.5) - x
            x = base + keeps * (x - base) + keeps * injected

        levels = G.channel_levels(G.to_unit(space.forward(base)))
        wanted = G.apply_grade(levels.reshape(1, 3, 1, 3),
                               grade.tonal()).reshape(levels.shape)
        got = G.channel_levels(G.to_unit(space.forward(x)))
        return float((got - levels).abs().mean() / (wanted - levels).abs().mean())

    results = {keeps: delivered(keeps) for keeps in (1.0, 0.8, 0.5)}
    assert results[1.0] > 0.95, results
    assert results[0.8] > 0.4, results      # the injection scheme managed 0.03
    assert results[0.5] > 0.2, results      # and 0.00
    assert results[1.0] > results[0.8] > results[0.5], results
    print("  grade delivered: " +
          ", ".join(f"{k:.0%} kept -> {v:.2f}" for k, v in results.items()))


def test_the_target_survives_the_detail_the_model_adds_later():
    """The first prediction inside the window is the finished picture without
    its fine detail, and fine detail carries colour: on an SDXL street the
    projection's chroma at 30 % was 0.83 of the finished picture's (0.70
    through the measured axes), so a target read there asked for less colour
    than the model was going to paint, and a hold with neutral wheels took
    17-30 % of the colour out. Read through a 4-pixel blur the two agree within
    2 %. Modelled here: a smooth picture, and the same picture with fine
    coloured detail on top; the chroma target must come out nearly the same.
    """
    N = _node_module()
    space = _space()
    torch.manual_seed(3)
    coarse = torch.nn.functional.interpolate(torch.randn(1, 4, 6, 6), size=(48, 48), mode="bilinear",
                                             align_corners=False)
    detail = torch.randn(1, 4, 48, 48) * 0.6
    grade = G.Grade(saturation=1.3)

    def target_of(latent):
        corrector = N.GradeCorrector(space, grade, 1.0)
        corrector(latent, 4.0)
        return float(corrector._target_chroma[0])

    raw = {k: float(G.chroma_level(G.to_unit(space.forward(v)))[0])
           for k, v in (("coarse", coarse), ("sharp", coarse + detail))}
    assert raw["sharp"] > raw["coarse"] * 1.3, raw             # the detail does add colour
    early, late = target_of(coarse), target_of(coarse + detail)
    assert abs(early / late - 1.0) < 0.15, (early, late, raw)
    print(f"  raw chroma {raw['coarse']:.3f} -> {raw['sharp']:.3f} with detail; "
          f"held target {early:.3f} vs {late:.3f}")


def test_measured_axes_replace_the_preview_matrix_for_the_wheels():
    """ComfyUI's latent_rgb_factors are a fit of the decoded picture, made for
    previews; inverted to write colour they are wrong by direction and size
    (SDXL: +red decodes as red minus green, darker, 3-5x). The wheels ask for
    the decoder's measured response instead, where the family has one."""
    from a1111_cc import axes, glue
    from a1111_cc import space as S

    class SDXL:                      # the latent format's class NAME is the key
        latent_channels = 4
        latent_dimensions = 2
        latent_rgb_factors = [[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068],
                              [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]]
        latent_rgb_factors_bias = [0.1084, -0.0175, -0.0011]

    class Model:
        def get_model_object(self, name):
            assert name == "latent_format"
            return SDXL()

    probe = torch.zeros(1, 4, 8, 8)
    preview = glue.detect_space(Model(), probe, "auto")
    measured = glue.detect_space(Model(), probe, "auto", measured_axes=True)
    assert preview.note != "measured axes" and preview.tone is None
    assert measured.note == "measured axes" and measured.tone is not None
    assert torch.allclose(measured._proj, torch.tensor(axes.MEASURED["SDXL"]["factors"]))

    # the reason: the preview fit is ill-conditioned, the measured axes are not
    def condition(space):
        sv = torch.linalg.svdvals(space._proj.double())
        return float(sv[0] / sv[-1])
    assert condition(preview) > 2 * condition(measured), (condition(preview), condition(measured))

    # unknown family: the preview matrix, with no tone curve
    class Mystery(SDXL):
        pass

    class Other:
        def get_model_object(self, name):
            return Mystery()
    fallback = glue.detect_space(Other(), probe, "auto", measured_axes=True)
    assert fallback.note != "measured axes" and fallback.tone is None
    print(f"  condition number: preview {condition(preview):.1f}, measured {condition(measured):.1f}")


def test_tone_response_asks_more_where_the_decoder_gives_less():
    """A latent colour change decodes at 0.45 of its size near black and 1.4
    in the midtones (SDXL, measured), so the level targets are divided by the
    response: twice the lift is asked for in the shadows."""
    N = _node_module()
    from a1111_cc import space as S
    torch.manual_seed(7)
    factors = torch.randn(4, 3) * 0.3
    plain = S.LatentColorSpace.rgb(factors, channels=4)
    toned = S.LatentColorSpace.rgb(factors, channels=4, tone={"v": [0.0, 0.5, 1.0], "r": [0.5, 1.0, 1.0]})
    grade = G.Grade(lift=G.Wheel(master=0.10))
    # a picture whose shadows sit near zero and midtones near a half
    x = torch.rand(1, 4, 32, 32) * 2.0 - 1.0
    deltas = {}
    for name, space in (("plain", plain), ("toned", toned)):
        corrector = N.GradeCorrector(space, grade, 1.0)
        corrector(x, 4.0)
        levels = G.channel_levels(corrector._smooth(G.to_unit(space.forward(x))))
        deltas[name] = corrector._targets - levels
    shadow = levels[0, :, 0].mean()
    ratio = float(deltas["toned"][0, :, 0].mean() / deltas["plain"][0, :, 0].mean())
    expected = 1.0 / N._ToneCurve({"v": [0.0, 0.5, 1.0], "r": [0.5, 1.0, 1.0]})(shadow)
    assert abs(ratio - expected) < 0.05, (ratio, expected, float(shadow))
    assert torch.allclose(deltas["toned"][0, :, 2], deltas["plain"][0, :, 2], atol=1e-5)  # highlights: r = 1
    print(f"  shadow level {float(shadow):.2f}: target delta x{ratio:.2f} (expected x{expected:.2f})")


def test_strength_is_how_much_of_the_grade():
    """strength was a rate: how fast the held target was approached. Over the
    eighteen or so calls of a real window every rate converges, so on SDXL
    0.25 and 1.0 gave nearly the same picture (colour x1.81 against x2.09 for
    Teal and orange) and the slider did nothing a user could see. It is now
    the amount: the target sits that fraction of the way from the ungraded
    levels to the graded ones. Measured here with a model that keeps all it is
    given, levels and colour separately, independently of the code under test.
    """
    N = _node_module()
    space = _space()
    grade = G.Grade(gain=G.Wheel(red=0.12), lift=G.Wheel(master=0.05))

    def levels_delivered(strength, calls=18):
        corrector = N.GradeCorrector(space, grade, strength)
        base = torch.randn(1, 4, 32, 32)
        x = base.clone()
        for i in range(calls):
            x = corrector(x, 10.0 - i * 0.5)
        levels = G.channel_levels(G.to_unit(space.forward(base)))
        wanted = G.apply_grade(levels.reshape(1, 3, 1, 3), grade.tonal()).reshape(levels.shape)
        got = G.channel_levels(G.to_unit(space.forward(x)))
        return float((got - levels).abs().mean() / (wanted - levels).abs().mean())

    def colour(latent):
        rgb = G.to_unit(space.forward(latent))
        grey = (rgb * torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        return float((rgb - grey).abs().mean())

    def colour_ratio(strength, calls=18):
        corrector = N.GradeCorrector(space, G.Grade(saturation=1.4), strength)
        x = torch.randn(1, 4, 32, 32)
        start = colour(x)
        for i in range(calls):
            x = corrector(x, 10.0 - i * 0.5)
        return colour(x) / start

    lv = {s: levels_delivered(s) for s in (0.25, 0.5, 1.0)}
    for s, got in lv.items():
        assert abs(got - s) < 0.1, f"strength {s} delivered {got:.2f} of the grade; {lv}"
    cr = {s: colour_ratio(s) for s in (0.25, 0.5, 1.0)}
    for s, got in cr.items():
        want = 1 + 0.4 * s
        assert abs(got - want) < 0.06, f"strength {s}: colour x{got:.3f}, wanted x{want:.2f}; {cr}"
    print("  levels delivered: " + ", ".join(f"{s} -> {v:.2f}" for s, v in lv.items())
          + "; colour (saturation 1.4): " + ", ".join(f"{s} -> x{v:.3f}" for s, v in cr.items()))


# --------------------------------------------------------------------------- #
# the presets
# --------------------------------------------------------------------------- #

def _presets():
    from a1111_cc import presets as P
    return P.BUILT_IN


def test_every_preset_is_a_sane_grade():
    """One of these shipped as a black frame.

    gain and gamma are absolute values around one, while lift is an offset
    from zero. Writing -0.06 for a gain, meaning "a little darker", asks for
    a gain of -0.06 instead of 0.94 — and the picture goes out.
    """
    out_of_range = []
    for name, values in _presets().items():
        for key in ("gain_master", "gamma_master"):
            if not 0.4 <= values[key] <= 1.6:
                out_of_range.append((name, key, values[key]))
        if not -0.3 <= values["lift_master"] <= 0.3:
            out_of_range.append((name, "lift_master", values["lift_master"]))
    assert not out_of_range, out_of_range


def test_no_preset_destroys_the_picture():
    """Applied to a ramp, every look must still be a picture."""
    N = _node_module()
    base = ramp()
    broken = []
    for name, values in _presets().items():
        graded = G.apply_grade(base, N._grade_from(values)).clamp(0, 1)
        mean = float(graded.mean())
        spread = float(graded.std())
        if not 0.1 < mean < 0.9 or spread < 0.1:
            broken.append((name, round(mean, 3), round(spread, 3)))
    assert not broken, broken


def test_every_preset_is_visibly_different_from_neutral():
    """A look nobody can see is not a look.

    Half of them were once so gentle that Teal and orange, Warm, Cool and
    Bleach bypass were indistinguishable from doing nothing at all.
    """
    N = _node_module()
    base = ramp()
    neutral = G.apply_grade(base, G.Grade())

    faint = []
    for name, values in _presets().items():
        if name == "Neutral":
            continue
        graded = G.apply_grade(base, N._grade_from(values)).clamp(0, 1)
        difference = float((graded - neutral).abs().mean())
        if difference < 0.02:
            faint.append((name, round(difference, 4)))
    assert not faint, faint
    print(f"  {len(_presets()) - 1} looks, all visible")


def test_presets_differ_from_each_other():
    """Two names for the same grade would be two names for nothing."""
    N = _node_module()
    base = ramp()
    rendered = {name: G.apply_grade(base, N._grade_from(values)).clamp(0, 1)
                for name, values in _presets().items()}

    names = sorted(rendered)
    duplicates = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if float((rendered[first] - rendered[second]).abs().mean()) < 0.01:
                duplicates.append((first, second))
    assert not duplicates, duplicates


def test_a_saved_name_keeps_single_spaces():
    """Characters outside the safe set are cut, and the spaces around them used
    to stay: «Тест — тёплые света» was saved as «Тест  тёплые света», two
    spaces, which a user cannot tell from one in the menu and cannot type back
    to delete it."""
    from a1111_cc import presets as P
    assert P.clean_name("Тест — тёплые света") == "Тест тёплые света"
    assert P.clean_name("  Warm · night  ") == "Warm night"
    assert P.clean_name("Day (v2) - soft") == "Day (v2) - soft", "the safe set itself is untouched"


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
