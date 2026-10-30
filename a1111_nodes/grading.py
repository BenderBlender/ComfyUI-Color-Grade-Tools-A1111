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
    on the targets it asks for nothing further, and `strength` means how fast
    to get there rather than how far to go.
    """

    def __init__(self, space, grade: G.Grade, strength: float, anchor=None):
        self.space = space
        self.grade = grade
        self.strength = strength
        self._anchor = anchor          # projected reference, or None
        self._targets = None
        self._injected = None
        self._last_sigma = None
        self._warned = set()

    wants_sigma = True

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
            self._injected = None
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

        With a reference the answer is fixed: grade *its* levels, once. Without
        one there is nothing to measure that the grade has not already touched
        — and the first prediction is no good as a stand-in, because early in
        sampling it is blurred and its spread is a fraction of the finished
        image's, so anchoring to it would squeeze contrast on every step.

        So the correction keeps track of what it has already injected and
        subtracts it back off, which recovers the levels the model would have
        produced on its own. If the model keeps our change, the estimate is
        exact; if it pushes back, we under-correct — the safe direction.
        """
        if self._anchor is not None:
            if self._targets is None:
                anchor = G.to_unit(self._anchor.to(unit.device, unit.dtype))
                self._targets = self._graded(G.channel_levels(anchor))
            return self._targets.to(unit.device, unit.dtype), None

        current = G.channel_levels(unit)
        if self._injected is None or self._injected.shape != current.shape:
            self._injected = torch.zeros_like(current)
        ungraded = current - self._injected.to(current.device, current.dtype)
        return self._graded(ungraded), current

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
            current = G.channel_levels(unit)
        if targets.shape[0] == 1 and current.shape[0] > 1:
            targets = targets.expand(current.shape[0], -1, -1)

        step = (targets - current) * self.strength
        mapped = G.piecewise_map(unit, current, current + step)
        if self._anchor is None:
            self._injected = self._injected + step

        if self.grade.saturation != 1.0:
            mapped = G.apply_grade(mapped, self.grade.per_pixel())

        delta = self.space.backward_delta(G.from_unit(mapped) - projected)
        if delta.shape != source.shape:
            self.warn_once("delta", "ColorKit Grade: correction came back as %s "
                                    "for a %s prediction; skipping.",
                           tuple(delta.shape), tuple(source.shape))
            return x0
        return (source + delta).to(x0.dtype)


class LatentColorGrade:
    """Grade while the image is being made, not after."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
            "strength": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "How quickly the prediction is brought to the graded "
                           "levels. The grade converges rather than piling up, so "
                           "this is a rate, not an amount: 1.0 snaps there on every "
                           "call, lower lets the model meet it half way.",
            }),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        if strength <= 0.0 or grade.is_neutral() or start_percent >= end_percent:
            return (glue.detach(model, key=key),)

        reference = model.get_model_object("latent_format")
        channels = getattr(reference, "latent_channels", 4) or 4
        probe = _probe_latent(model, channels)
        space = glue.detect_space(model, probe, "auto")
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
