"""Colour grading with the three wheels of a grading suite.

Two nodes, one grade. ``Latent Color Grade`` patches the model so the grade
is applied to every prediction while the image is being made — the model then
composes *in* the graded look rather than having it laid over the top
afterwards. ``Image Color Grade`` does the same arithmetic to a decoded image,
for when a straightforward post-pass is what is wanted.

This supersedes the Recenter nodes from ``comfyui-diffusion-cg`` (MIT,
Haoming02; archived July 2026, and its own README lists "does not work with
certain samplers" as a known issue). Three things are different here:

* **Nothing is global.** Recenter monkey-patched ``comfy.sample.sample`` and
  kept its settings in module-level variables, which is why its README tells
  you to add a second node with strength 0 to switch the effect off again.
  The grade lives on the MODEL wire, so it reaches exactly the sampler it is
  plugged into.
* **Every sampler works.** Recenter wrote into the sampler's working tensor
  from a callback, which several samplers discard. This hooks the noise
  prediction instead, which all of them use.
* **The controls are colours, not channels.** Recenter mapped sliders onto
  raw latent channels, so it needed a different node per model family and
  none at all for the rest. Going through the model's own latent-to-RGB
  projection means lift, gamma and gain mean the same thing on SD1.5, SDXL,
  SD3, Flux and the video models.
"""

import logging

import torch

from ..a1111_cc import glue
from ..a1111_cc import grading as G

log = logging.getLogger(__name__)

# Our own wrapper key: detaching must not lift someone else's hook.
GRADE_KEY = "a1111_color_grade"

WHEELS = ("lift", "gamma", "gain")
CHANNELS = ("r", "g", "b")


def _wheel_inputs(name: str, centre: float, tooltip: str) -> dict:
    """The four numbers behind one wheel: a chroma offset per channel, plus a
    master. The colour-wheel widget writes into these, and they stay usable on
    their own if the widget is not loaded."""
    step = 0.005
    span = 0.5
    entries = {}
    for channel in CHANNELS:
        entries[f"{name}_{channel}"] = ("FLOAT", {
            "default": 0.0, "min": -span, "max": span, "step": step,
            "tooltip": f"{tooltip} — {channel.upper()} offset.",
        })
    entries[f"{name}_master"] = ("FLOAT", {
        "default": centre, "min": centre - span, "max": centre + span, "step": step,
        "tooltip": f"{tooltip} — all three channels together.",
    })
    return entries


def _grade_from(values: dict) -> G.Grade:
    def wheel(name, centre=0.0):
        return G.Wheel(
            red=float(values.get(f"{name}_r", 0.0)),
            green=float(values.get(f"{name}_g", 0.0)),
            blue=float(values.get(f"{name}_b", 0.0)),
            master=float(values.get(f"{name}_master", centre)) - centre,
        )

    grade = G.Grade(
        lift=wheel("lift"),
        gamma=wheel("gamma", 1.0),
        gain=wheel("gain", 1.0),
        saturation=float(values.get("saturation", 1.0)),
        contrast=float(values.get("contrast", 1.0)),
    )
    # Wheel.master is an offset from neutral and as_tensor adds the centre
    # back on, so nothing more is needed here. Adding the centre a second time
    # is what made an untouched node grade at gamma 2 and gain 2.
    return grade


GRADE_CONTROLS = {}
GRADE_CONTROLS.update(_wheel_inputs("lift", 0.0, "Shadows, pivoting on white"))
GRADE_CONTROLS.update(_wheel_inputs("gamma", 1.0, "Midtones"))
GRADE_CONTROLS.update(_wheel_inputs("gain", 1.0, "Highlights, pivoting on black"))
GRADE_CONTROLS["saturation"] = ("FLOAT", {
    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
    "tooltip": "0 is monochrome, 1 leaves saturation alone.",
})
GRADE_CONTROLS["contrast"] = ("FLOAT", {
    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
    "tooltip": "Pivots on mid grey.",
})


class GradeCorrector:
    """Applies a grade to each prediction, in the model's colour subspace.

    The targets are worked out **once** and then held. That is what stops the
    grade compounding: the hook runs on every call the sampler makes, so a
    correction expressed as "shift by this much" would be applied twenty or
    fifty times over, and the result would depend on the step count and on
    whether the sampler evaluates the model once or twice per step. Expressed
    as "put the shadows here", it converges — once the prediction already sits
    on the targets it asks for nothing further.

    `strength` is how far to go: the targets sit that fraction of the way from
    the ungraded levels to the graded ones. Each call closes a fixed share of
    what is left (RATE). strength used to be that share, and over the eighteen
    or so calls of a real window every share converges, so on SDXL 0.25 and 1.0
    gave nearly the same picture.

    Levels and colour are read off a box-blurred projection (BLUR). The first
    prediction inside the window, where the targets are taken from, is the
    finished picture without its fine detail, and fine detail carries colour:
    on an SDXL street the projection's chroma at 30 % of the schedule was 0.83
    of the finished one (0.70 through the measured axes), so a target taken
    there asked for LESS colour than the model was going to paint anyway, and
    a hold with neutral wheels took 17-30 % of the colour out of the finished
    picture. After a 4-pixel blur the same chroma is 1.02 / 0.98 of the
    finished picture's and the hold takes out 2 %; the levels' spread agrees
    within 5 %. The map built from blurred levels is applied to the full
    picture; for lift, gain and contrast it is the same map, and the colour
    factor is the same ratio.

    When the space carries the decoder's tonal response (measured axes), the
    level targets are divided by it: a latent colour change at a shadow value
    decodes at about 0.45 of its size, in the midtones at 1.4, so the
    projection has to be asked for more in the shadows and less in the
    midtones for the decoded picture to move by the grade.
    """

    RATE = 0.5      # share of the remaining distance closed per call; the old default
    BLUR = 5        # latent pixels; measured 4-6 keep the chroma within 2 % along the run

    def __init__(self, space, grade: G.Grade, strength: float, anchor=None):
        self.space = space
        self.grade = grade
        self.strength = strength
        self._anchor = anchor          # projected reference, or None
        self._targets = None
        self._target_chroma = None     # held like the levels, worked out with them
        self._last_sigma = None
        self._warned = set()
        tone = getattr(space, "tone", None)
        self._tone = _ToneCurve(tone) if tone else None

    wants_sigma = True

    def _smooth(self, unit):
        """The projection without its fine detail, for measuring levels and colour."""
        k = self.BLUR
        if k <= 1:
            return unit
        pad = k // 2
        if unit.ndim == 5:
            return torch.nn.functional.avg_pool3d(unit, (1, k, k), stride=1, padding=(0, pad, pad),
                                                  count_include_pad=False)
        return torch.nn.functional.avg_pool2d(unit, k, stride=1, padding=pad, count_include_pad=False)

    def _note_sigma(self, sigma):
        """Forget what we learned when a new run starts.

        Sigma falls through a run. If it jumps back up, the sampler has begun
        again — and ComfyUI hands back the same patched model whenever the
        node's inputs have not changed, so without this the targets worked out
        for one seed would quietly steer the next.
        """
        if sigma is None:
            return
        if self._last_sigma is not None and sigma > self._last_sigma * 1.05:
            self._targets = None
        self._last_sigma = sigma

    def warn_once(self, key, msg, *args):
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _graded(self, levels):
        shaped = levels.reshape(levels.shape[0], levels.shape[1], 1, 3)
        return G.apply_grade(shaped, self.grade.tonal()).reshape(levels.shape)

    def _target_levels(self, unit):
        """Where the shadows, midtones and highlights should end up.

        Worked out once and then held. With a reference the targets come from
        it; without one they are taken from the first prediction inside the
        window. At the top of the schedule that is a blur whose spread is a
        fraction of the finished image's, so the window does not start there.
        Starting a third of the way in was not late enough either: a hold
        taken from there also resists everything the model repaints
        afterwards, and that resistance is measured as part of the grade. On
        an SDXL img2img at denoise 0.7 the hold alone (wheels a hair from
        neutral) shifted the colour 2.7 times as far as Teal and orange applied
        afterwards; the grade from 0.3 came out at 3.4 times, from 0.6 at 1.1,
        while a txt2img street stayed at 0.9 either way. Hence 0.6.

        An earlier version tried to estimate the ungraded image instead, by
        tracking what it had already injected and subtracting it back off.
        That is exact only if the model keeps every bit of the correction. It
        does not: simulated against a model that pushes back twenty percent
        per call, that scheme delivered three percent of the grade, and at
        fifty percent pushback none of it. Holding a fixed target degrades
        gently instead — 0.82, 0.67 and 0.33 of the grade for the same three
        cases — because every call pulls towards the same place rather than
        concluding the work is already done.
        """
        if self._targets is None:
            if self._anchor is not None:
                base = G.to_unit(self._anchor.to(unit.device, unit.dtype))
            else:
                base = unit
            base = self._smooth(base)
            levels = G.channel_levels(base)
            if self.grade.saturation == 1.0:
                # per-channel curves commute with percentiles: grade three numbers
                graded = self._graded(levels)
            else:
                # Saturation mixes channels, so grade the base itself and read both
                # targets off that one picture. Taking the levels from an unsaturated
                # base and the colour from a saturated one gave two targets that
                # disagreed — the curve pulled the channels together, the colour
                # pushed them apart, and the correction never settled.
                picture = G.apply_grade(base, self.grade)
                graded = G.channel_levels(picture)
                chroma = G.chroma_level(base)
                self._target_chroma = chroma + (G.chroma_level(picture) - chroma) * self.strength
            wanted = (graded - levels) * self.strength
            if self._tone is not None:
                # The decoder lets r(v) of a latent colour change through at tone v
                # (0.45 near black, 1.4 in the midtones on SDXL), so the projection
                # must move by 1/r of the grade for the picture to move by the grade.
                # Applied to the chroma target as well this over-corrected (the
                # colour factor landed at 0.6 of the image grade's), so it is not.
                wanted = wanted / self._tone(levels)
            self._targets = levels + wanted
        return self._targets.to(unit.device, unit.dtype), None

    def __call__(self, x0, sigma=None):
        self._note_sigma(sigma)
        if x0.ndim not in (4, 5) or x0.shape[1] != self.space.channels:
            self.warn_once("shape", "ColorKit Grade: skipping a %s prediction",
                           tuple(x0.shape))
            return x0

        source = x0.float()
        projected = self.space.forward(source)
        unit = G.to_unit(projected)

        targets, current = self._target_levels(unit)
        if current is None:
            current = G.channel_levels(self._smooth(unit))
        if targets.shape[0] == 1 and current.shape[0] > 1:
            targets = targets.expand(current.shape[0], -1, -1)

        step = (targets - current) * self.RATE
        mapped = G.piecewise_map(unit, current, current + step)

        if self.grade.saturation != 1.0:
            # Saturation used to be applied here as a factor on every call, and it
            # was the one control that compounded: on SDXL, 26 steps from 30% of the
            # schedule, 1.15 raised the finished image's colour 3.2-fold. It now
            # pulls towards a held target, like the levels above.
            current = G.chroma_level(self._smooth(mapped))
            target = self._target_chroma.to(current.device, current.dtype)
            if target.shape[0] == 1 and current.shape[0] > 1:
                target = target.expand(current.shape[0])
            wanted = current + (target - current) * self.RATE
            factor = (wanted / current.clamp_min(1e-6)).clamp(0.0, 2.0)
            mapped = G.scale_chroma(mapped, factor)

        delta = self.space.backward_delta(G.from_unit(mapped) - projected)
        if delta.shape != source.shape:
            self.warn_once("delta", "ColorKit Grade: correction came back as %s "
                                    "for a %s prediction; skipping.",
                           tuple(delta.shape), tuple(source.shape))
            return x0
        return (source + delta).to(x0.dtype)


class _ToneCurve:
    """Piecewise-linear r(v) from a measured table, evaluated on any tensor of unit values."""

    def __init__(self, table):
        self.v = torch.tensor(table["v"], dtype=torch.float32)
        self.r = torch.tensor(table["r"], dtype=torch.float32).clamp_min(0.3)

    def __call__(self, value):
        v = self.v.to(value.device, value.dtype)
        r = self.r.to(value.device, value.dtype)
        x = value.clamp(0.0, 1.0)
        idx = torch.bucketize(x, v).clamp(1, len(v) - 1)
        lo, hi = v[idx - 1], v[idx]
        w = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        return r[idx - 1] * (1.0 - w) + r[idx] * w


class LatentColorGrade:
    """Grade while the image is being made, not after."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "How much of the grade: 1.0 the full look, 0.5 half of it. "
                           "The grade is held as a target and approached a little on "
                           "every step, so it neither piles up nor depends on the "
                           "step count.",
            }),
            "start_percent": ("FLOAT", {
                "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Where in the schedule to begin. Without an anchor the "
                           "grade takes its bearings from the first prediction it "
                           "sees and holds those levels, so it also resists whatever "
                           "the model repaints afterwards. Earlier gives the model "
                           "longer to compose in the look but lets that hold show — "
                           "on a denoise-0.7 img2img, 0.3 came out 3.4 times the "
                           "grade, 0.6 at 1.1 times.",
            }),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        required.update(GRADE_CONTROLS)
        return {
            "required": required,
            "optional": {
                "anchor": ("LATENT", {
                    "tooltip": "Optional. With a latent here the grade aims at its "
                               "levels instead of the model's own — the img2img source "
                               "is the natural choice. Without it the correction works "
                               "out its own target as it goes.",
                }),
            },
            # Each node gets its own hook, so two grades in a chain stack
            # instead of the second quietly lifting the first.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "ColorKit/model"
    DESCRIPTION = (
        "Lift, gamma and gain applied to the model's own prediction, so the image is "
        "composed in the graded look. Works with any sampler and any model family."
    )

    def apply(self, model, strength, start_percent, end_percent,
              anchor=None, unique_id=None, **controls):
        key = f"{GRADE_KEY}:{unique_id}" if unique_id is not None else GRADE_KEY
        grade = _grade_from(controls)
        # Neutral wheels with an anchor still do something — they aim the
        # prediction at the anchor's levels — so only neutral AND unanchored is
        # a no-op. ImageColorGrade has always tested its reference the same way.
        if (strength <= 0.0 or start_percent >= end_percent
                or (grade.is_neutral() and anchor is None)):
            return (glue.detach(model, key=key),)

        reference = model.get_model_object("latent_format")
        channels = getattr(reference, "latent_channels", 4) or 4
        probe = _probe_latent(model, channels)
        # The wheels invert the matrix to write colour; the preview fit inverts
        # badly (SDXL: magenta, 3-5x too strong), so the measured axes are asked for.
        space = glue.detect_space(model, probe, "auto", measured_axes=True)
        if space.kind == "raw":
            log.warning("Latent Color Grade: this model has no latent-to-RGB matrix, so "
                        "the wheels have no colour axes to work in; leaving it unpatched.")
            return (glue.detach(model, key=key),)

        projected_anchor = None
        if anchor is not None:
            samples = glue.fit_reference_dims(model, anchor["samples"])
            projected_anchor = space.forward(glue.to_model_space(model, samples).float())

        sigma_hi, sigma_lo = glue.sigma_window(model, start_percent, end_percent)
        corrector = GradeCorrector(space, grade, float(strength), projected_anchor)
        return (glue.attach(model, corrector, sigma_hi, sigma_lo, key=key),)


def _probe_latent(model, channels):
    """A stand-in latent, only used so the space can be detected up front."""
    import torch
    latent_format = model.get_model_object("latent_format")
    dims = getattr(latent_format, "latent_dimensions", 2)
    shape = [1, channels, 8, 8] if dims == 2 else [1, channels, 1, 8, 8]
    return torch.zeros(shape)


class ImageColorGrade:
    """The same grade, applied to a decoded image."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {"image": ("IMAGE",)}
        required.update(GRADE_CONTROLS)
        return {
            "required": required,
            "optional": {
                "reference": ("IMAGE", {
                    "tooltip": "Optional. With a picture here the wheels stop being "
                               "absolute and start from its look instead: the image is "
                               "mapped onto the reference's shadows, midtones and "
                               "highlights, and the wheels then move those. Leave it "
                               "empty to grade the image on its own terms.",
                }),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "ColorKit/image"
    DESCRIPTION = (
        "Lift, gamma and gain on an image, with the usual colourist's controls. "
        "Connect a reference picture to pick up its look and adjust from there."
    )

    def apply(self, image, reference=None, mask=None, **controls):
        import torch

        grade = _grade_from(controls)
        if grade.is_neutral() and reference is None:
            return (image,)

        source = image.detach().float().movedim(-1, 1)[:, :3].clamp(0.0, 1.0)
        if reference is None:
            graded = G.apply_grade(source, grade)
        else:
            anchor = reference.detach().float().movedim(-1, 1)[:, :3].clamp(0.0, 1.0)
            graded = G.grade_levels(source, grade, anchor)
        graded = graded.clamp(0.0, 1.0)

        if mask is not None:
            weights = mask.detach().float()
            if weights.ndim == 2:
                weights = weights.unsqueeze(0)
            weights = torch.nn.functional.interpolate(
                weights.unsqueeze(1), size=source.shape[-2:],
                mode="bilinear", align_corners=False)
            if weights.shape[0] != source.shape[0]:
                weights = weights[:1].expand(source.shape[0], -1, -1, -1)
            graded = source + (graded - source) * weights.clamp(0.0, 1.0)

        out = graded.movedim(1, -1)
        if image.shape[-1] == 4:
            out = torch.cat([out, image[..., 3:]], dim=-1)
        return (out.to(device=image.device, dtype=image.dtype),)


NODE_CLASS_MAPPINGS = {
    "LatentColorGrade": LatentColorGrade,
    "ImageColorGrade": ImageColorGrade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentColorGrade": "ColorKit Grade wheels — latent (model patch)",
    "ImageColorGrade": "ColorKit Grade wheels — image",
}
