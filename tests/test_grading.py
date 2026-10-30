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


def test_the_order_of_operations_is_the_documented_one():
    """lift first, then gain, then gamma — and each sees the previous result.

    Gain multiplying what lift has already raised is the point: that is the
    order a grading panel works in, and it is why the two controls feel
    different rather than interchangeable.
    """
    base = ramp()
    lift = G.Wheel(master=0.15)
    gain = G.Wheel(master=0.2)

    lifted = G.apply_grade(base, G.Grade(lift=lift))
    both = G.apply_grade(base, G.Grade(lift=lift, gain=gain))

    expected = lifted * 1.2
    assert torch.allclose(both, expected, atol=1e-5), \
        float((both - expected).abs().max())

    # And applied alone, each still leaves the far end of the ramp alone.
    assert abs(float(lifted[..., -1].mean() - base[..., -1].mean())) < 1e-5
    gained = G.apply_grade(base, G.Grade(gain=gain))
    assert abs(float(gained[..., 0].mean() - base[..., 0].mean())) < 1e-5
    print("  lift -> gain -> gamma, each seeing the last")


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
