"""Lift / gamma / gain colour grading — the three wheels of a grading suite.

The controls are the ones a colourist expects, and they do what they do in
Resolve or Baselight:

* **lift** moves the shadows and leaves white where it is;
* **gain** scales the highlights and leaves black where it is;
* **gamma** bends the midtones between them;
* **offset** moves everything together.

Each is an RGB triple — that is what a colour wheel is: a point on a disc,
read as a pair of chroma offsets, plus a master ring for the achromatic part.

Applied in order, per channel::

    v = v + offset                       everything together
    v = v * (gain - lift) + lift         black lands on lift, white on gain
    v = v ** (1 / gamma)                 midtones bend between them

Written that way the two endpoints are independent — gain cannot drag the
black point around once lift has moved it — and the pair is exactly the
slope/offset of ASC CDL, so a grade can be exported.

Values are graded in [0, 1], which is where those formulas are defined. The
latent projection this is fed from lives in roughly [-1, 1] — the same range
ComfyUI's own previewer maps to an image — so :func:`to_unit` and
:func:`from_unit` move between the two.

Pure torch; nothing here imports ComfyUI.
"""

from dataclasses import dataclass, field
from typing import List

import torch

EPS = 1e-6
GAMMA_FLOOR = 0.01
# How far a level-based map may stretch or squash the image, and the spread
# below which it stops trying to and just shifts.
MAX_SLOPE = 4.0
FLAT_RANGE = 0.02


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Projected-RGB range to [0, 1], the same mapping the previewer uses."""
    return (x + 1.0) * 0.5


def from_unit(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


@dataclass
class Wheel:
    """One colour wheel: a chroma offset per channel, plus a master."""

    red: float = 0.0
    green: float = 0.0
    blue: float = 0.0
    master: float = 0.0

    def as_tensor(self, like: torch.Tensor, centre: float = 0.0) -> torch.Tensor:
        """``[1, 3, 1, 1]`` (or 5-D) of per-channel values, master included."""
        values = [centre + self.master + self.red,
                  centre + self.master + self.green,
                  centre + self.master + self.blue]
        shape = [1, 3] + [1] * (like.ndim - 2)
        return torch.tensor(values, device=like.device, dtype=like.dtype).reshape(shape)

    def is_neutral(self) -> bool:
        """True when this wheel changes nothing. ``master`` is an offset from
        neutral, so zero is neutral whatever the centre happens to be."""
        return self.red == self.green == self.blue == self.master == 0.0


@dataclass
class Grade:
    """A full grade: the three wheels plus a shared offset."""

    lift: Wheel = field(default_factory=Wheel)
    gamma: Wheel = field(default_factory=Wheel)
    gain: Wheel = field(default_factory=Wheel)
    offset: Wheel = field(default_factory=Wheel)
    saturation: float = 1.0
    contrast: float = 1.0
    pivot: float = 0.5

    def is_neutral(self) -> bool:
        return (self.lift.is_neutral() and self.gain.is_neutral()
                and self.gamma.is_neutral() and self.offset.is_neutral()
                and self.saturation == 1.0 and self.contrast == 1.0)

    def tonal(self) -> "Grade":
        """The part that can be expressed as a curve on each channel.

        Saturation cannot: it mixes the channels together, so running it
        through three percentiles would pair the fifth percentile of red with
        the fifth percentile of green — different pixels entirely. It is
        applied to the image itself instead, after the curve.
        """
        return Grade(lift=self.lift, gamma=self.gamma, gain=self.gain,
                     offset=self.offset, contrast=self.contrast, pivot=self.pivot)

    def per_pixel(self) -> "Grade":
        return Grade(saturation=self.saturation)


# Rec.709 luma, for the saturation control
_LUMA = (0.2126, 0.7152, 0.0722)


def apply_grade(rgb: torch.Tensor, grade: Grade) -> torch.Tensor:
    """Grade an RGB tensor already in [0, 1]. Channel axis is 1."""
    out = rgb

    offset = grade.offset.as_tensor(rgb)
    if float(offset.abs().max()) > 0:
        out = out + offset

    lift = grade.lift.as_tensor(rgb)
    gain = grade.gain.as_tensor(rgb, centre=1.0)
    if float(lift.abs().max()) > 0 or float((gain - 1.0).abs().max()) > 0:
        # The classic form: black lands on lift, white lands on gain. Written
        # as `v + lift*(1-v)` followed by `v*gain` instead, gain would drag
        # the black point along with it whenever lift was non-zero, and the
        # two controls would stop being independent. This is also the slope /
        # offset pair of ASC CDL, which makes the grade exportable.
        out = out * (gain - lift) + lift

    gamma = grade.gamma.as_tensor(rgb, centre=1.0).clamp_min(GAMMA_FLOOR)
    if float((gamma - 1.0).abs().max()) > 0:
        inverse = 1.0 / gamma
        # Below zero and above one a power curve is either undefined or wildly
        # steep. Negatives pass straight through, as ASC CDL specifies, and
        # above white the curve continues along its own tangent. Both matter
        # in the latent, where a few percent of the projection sits outside
        # [0, 1] — clamping them instead would crush the darkest pixels onto
        # black a little more on every sampling step.
        powered = out.clamp(0.0, 1.0) ** inverse
        extended = 1.0 + (out - 1.0) * inverse
        out = torch.where(out > 1.0, extended, torch.where(out > 0.0, powered, out))

    if grade.contrast != 1.0:
        out = (out - grade.pivot) * grade.contrast + grade.pivot

    if grade.saturation != 1.0:
        weights = torch.tensor(_LUMA, device=rgb.device, dtype=rgb.dtype)
        weights = weights.reshape([1, 3] + [1] * (rgb.ndim - 2))
        luma = (out * weights).sum(dim=1, keepdim=True)
        out = luma + (out - luma) * grade.saturation

    return out


def grade_unit_range(rgb: torch.Tensor, grade: Grade) -> torch.Tensor:
    """Grade a tensor in the projection's own range, converting both ways."""
    return from_unit(apply_grade(to_unit(rgb), grade))


# --------------------------------------------------------------------------- #
# grading towards a target, rather than by an offset
# --------------------------------------------------------------------------- #

LEVELS = (0.05, 0.50, 0.95)


def channel_levels(rgb: torch.Tensor, levels=LEVELS) -> torch.Tensor:
    """Where this image's shadows, midtones and highlights actually sit.

    Returns ``[B, C, len(levels)]`` — a percentile per channel, which is what
    "the shadows of this frame" means in practice. Using percentiles rather
    than the extremes keeps a handful of stray pixels from defining the range.
    """
    flat = rgb.reshape(rgb.shape[0], rgb.shape[1], -1).float()
    quantiles = torch.tensor(levels, device=flat.device, dtype=flat.dtype)
    # quantile is limited in how many elements it will take, so sample when big
    if flat.shape[-1] > 1_000_000:
        step = flat.shape[-1] // 500_000
        flat = flat[..., ::step]
    return torch.quantile(flat, quantiles, dim=-1).permute(1, 2, 0)


def piecewise_map(rgb: torch.Tensor, source: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    """Move ``source`` levels onto ``target`` levels, smoothly in between.

    Two straight segments joined at the midtone, each continuing past the
    outer knots along its own slope. Every level the grade was measured at
    lands exactly on its target, so applying this repeatedly converges instead
    of compounding: once the image already sits on the targets, it asks for no
    further change.
    """
    batch = rgb.shape[0]
    if source.shape[0] == 1 and batch > 1:
        source = source.expand(batch, -1, -1)
    if target.shape[0] == 1 and batch > 1:
        target = target.expand(batch, -1, -1)

    shape = [batch, rgb.shape[1]] + [1] * (rgb.ndim - 2)
    s0, s1, s2 = (source[..., i].reshape(shape) for i in range(3))
    t0, t1, t2 = (target[..., i].reshape(shape) for i in range(3))

    lower = ((t1 - t0) / (s1 - s0).clamp_min(EPS)).clamp(1.0 / MAX_SLOPE, MAX_SLOPE)
    upper = ((t2 - t1) / (s2 - s1).clamp_min(EPS)).clamp(1.0 / MAX_SLOPE, MAX_SLOPE)

    # A prediction early in sampling is nearly flat, and so is a plain colour
    # field. Dividing by that range would multiply contrast by a hundred, so
    # below a usable spread the map becomes a pure shift: move the midtone
    # where it was asked to go and leave the shape of the image alone.
    spread = (s2 - s0)
    flat = spread < FLAT_RANGE
    lower = torch.where(flat, torch.ones_like(lower), lower)
    upper = torch.where(flat, torch.ones_like(upper), upper)

    # Both halves are anchored at the midtone rather than at their outer knot.
    # Unclamped the two are identical — the lower segment still passes exactly
    # through (s0, t0). Clamped they are not: anchoring each end separately
    # would leave the two halves arriving at different values either side of
    # the midtone, a step in the middle of the curve that shows up as a hard
    # edge through the midtones. This way the map stays continuous and simply
    # falls short of the outer targets, which is the honest failure.
    below = t1 + (rgb - s1) * lower
    above = t1 + (rgb - s1) * upper
    mapped = torch.where(rgb < s1, below, above)
    return torch.where(flat, rgb + (t1 - s1), mapped)


def grade_levels(rgb: torch.Tensor, grade: Grade,
                 anchor: torch.Tensor = None) -> torch.Tensor:
    """Grade by moving this image's levels onto graded ones, all in [0, 1].

    With an ``anchor`` the targets are measured from *that* picture, so the
    result takes the anchor's shadows, midtones and highlights — and the
    wheels then move those targets rather than the image's own. That is the
    difference between "make it look like this" and "make it look like this,
    but warmer", and it costs nothing extra: the grade is applied to three
    numbers per channel, not to every pixel.
    """
    source = channel_levels(rgb)
    base = channel_levels(anchor if anchor is not None else rgb)

    shaped = base.reshape(base.shape[0], base.shape[1], 1, 3)
    graded = apply_grade(shaped, grade.tonal()).reshape(base.shape)

    mapped = piecewise_map(rgb, source, graded)
    if grade.saturation != 1.0:
        mapped = apply_grade(mapped, grade.per_pixel())
    return mapped


def grade_towards(rgb: torch.Tensor, grade: Grade,
                  anchor: torch.Tensor = None) -> torch.Tensor:
    """:func:`grade_levels` for tensors in the latent projection's range."""
    base = to_unit(anchor) if anchor is not None else None
    return from_unit(grade_levels(to_unit(rgb), grade, base))


def wheel_from_polar(angle_degrees: float, radius: float, master: float = 0.0,
                     gain: float = 1.0) -> Wheel:
    """Build a wheel from where a handle sits on the disc.

    The disc is the usual colourist's layout: red to the right, green at 120
    degrees counter-clockwise, blue at 240. A handle at the centre is neutral,
    and the radius sets how far the cast goes.
    """
    import math

    angle = math.radians(angle_degrees)
    components: List[float] = []
    for channel in range(3):
        channel_angle = math.radians(channel * 120.0)
        components.append(radius * gain * math.cos(angle - channel_angle))

    # Taking the mean out keeps the wheel purely chromatic; brightness is the
    # master's job, so the two controls stay independent.
    mean = sum(components) / 3.0
    return Wheel(red=components[0] - mean,
                 green=components[1] - mean,
                 blue=components[2] - mean,
                 master=master)
