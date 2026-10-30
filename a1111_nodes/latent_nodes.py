"""Latent-space colour matching: a model patch, and a one-shot LATENT node."""

import logging

from ..a1111_cc import glue
from ..a1111_cc.guidance import GuidanceConfig, LatentCorrector
from ..a1111_cc.space import LatentColorSpace

log = logging.getLogger(__name__)

METHODS = ["auto", "mean_only", "reinhard", "mkl"]
SPACES = ["auto", "rgb_proj", "raw"]
TEMPORAL = ["global", "per_frame"]

_METHOD_TOOLTIP = (
    "auto: mean_only when the latent's channels are not colour axes, reinhard otherwise.\n"
    "mean_only: shift each channel's average — cannot touch contrast or detail.\n"
    "reinhard: match average and spread per channel.\n"
    "mkl: optimal linear transport; also corrects casts that lie across channels, "
    "at the cost of a small eigendecomposition per step."
)
_SPACE_TOOLTIP = (
    "auto/rgb_proj: project through the model's own latent-to-RGB matrix, so the "
    "correction only moves the three directions that carry colour. Works for every "
    "model family.\n"
    "raw: match the latent channels directly — faithful to the original node, but "
    "only meaningful for 4-channel SD/SDXL latents."
)


def _resolve_method(requested: str, space: LatentColorSpace) -> str:
    return space.default_method if requested == "auto" else requested


class ApplyLatentColorGuidance:
    """Hold a model's colours to a reference while it samples."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_latent": ("LATENT", {
                    "tooltip": "The latent whose colours the result should take. "
                               "Usually VAEEncode of the image you are matching to.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the correction to apply at each model call.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where in the schedule to start. The first few percent are "
                               "left alone on purpose: the prediction is still almost "
                               "featureless there and its spread is not yet meaningful.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where to stop. Ending before the last steps lets the model "
                               "settle detail without the correction nudging it.",
                }),
                "method": (METHODS, {"default": "auto", "tooltip": _METHOD_TOOLTIP}),
                "space": (SPACES, {"default": "auto", "tooltip": _SPACE_TOOLTIP}),
                "max_gain": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Ceiling on the per-channel scale, so a nearly flat "
                               "prediction cannot be amplified into artefacts.",
                }),
                "lowpass": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Blur the correction over this radius in latent pixels, "
                               "keeping the colour shift and dropping the detail "
                               "sharpening that a gain above 1 brings with it.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Measure the colours only inside this area — useful for "
                               "inpainting, where the surrounding plate would otherwise "
                               "drag the statistics towards doing nothing.",
                }),
                "temporal": (TEMPORAL, {
                    "default": "global",
                    "tooltip": "Video latents only. global applies one correction to the "
                               "whole clip and cannot flicker; per_frame tracks drift "
                               "within the clip but can.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/sampling"
    DESCRIPTION = (
        "Patch a model so that, while it samples, each prediction is pulled towards a "
        "reference latent's colours. Works with any sampler and any model family."
    )

    def apply(self, model, reference_latent, strength, start_percent, end_percent,
              method, space, max_gain, lowpass, mask=None, temporal="global"):
        if strength <= 0.0 or start_percent >= end_percent:
            if start_percent >= end_percent:
                log.warning(
                    "ApplyLatentColorGuidance: start_percent (%.2f) is not before "
                    "end_percent (%.2f); leaving the model unpatched.",
                    start_percent, end_percent,
                )
            # Turning the node down to zero should also lift a hook applied
            # further up the graph, which is what "off" looks like to a user.
            return (glue.detach(model),)

        reference = glue.fit_reference_dims(model, reference_latent["samples"])
        colour_space = glue.detect_space(model, reference, space)
        reference = glue.to_model_space(model, reference)

        config = GuidanceConfig(
            method=_resolve_method(method, colour_space),
            strength=float(strength),
            max_gain=float(max_gain),
            lowpass=int(lowpass),
            temporal=temporal,
        )
        # The mask is handed over unscaled: the corrector fits it to whatever
        # is actually being sampled, which is not always the reference's size.
        corrector = LatentCorrector(colour_space, reference, config, mask)

        sigma_hi, sigma_lo = glue.sigma_window(model, start_percent, end_percent)
        log.info("A1111 latent colour: %s, method=%s, sigma %.4f..%.4f",
                 colour_space, config.method, sigma_lo, sigma_hi)
        return (glue.attach(model, corrector, sigma_hi, sigma_lo),)


class LatentColorMatch:
    """Match one latent's colours to another, in one shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "reference": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (METHODS, {"default": "auto", "tooltip": _METHOD_TOOLTIP}),
                "space": (SPACES, {"default": "auto", "tooltip": _SPACE_TOOLTIP}),
                "max_gain": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Optional, but recommended: without it there is no latent "
                               "format to read, so the match falls back to raw channels "
                               "instead of the RGB projection.",
                }),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "apply"
    CATEGORY = "latent/advanced"
    DESCRIPTION = (
        "Pull a latent's colours onto a reference latent in a single step — before "
        "sampling to steer an img2img pass, or after it to correct the result."
    )

    def apply(self, samples, reference, strength, method, space, max_gain,
              model=None, mask=None):
        out = samples.copy()
        if strength <= 0.0:
            return (out,)

        source = samples["samples"]
        ref = reference["samples"]

        if model is None:
            if space == "rgb_proj":
                raise ValueError(
                    "space='rgb_proj' needs the MODEL input, to read its latent format."
                )
            colour_space = LatentColorSpace.raw(
                ref.shape[1], source.ndim,
                default_method="reinhard" if ref.shape[1] <= 4 else "mean_only",
            )
            config = GuidanceConfig(
                method=_resolve_method(method, colour_space),
                strength=float(strength),
                max_gain=float(max_gain),
            )
            # Both latents are in VAE space and so already on the same scale as
            # each other, which is all a raw per-channel match needs.
            out["samples"] = LatentCorrector(colour_space, ref, config, mask)(source)
            return (out,)

        ref = glue.fit_reference_dims(model, ref)
        source = glue.fit_reference_dims(model, source)
        colour_space = glue.detect_space(model, ref, space)
        config = GuidanceConfig(
            method=_resolve_method(method, colour_space),
            strength=float(strength),
            max_gain=float(max_gain),
        )
        # The RGB projection is defined against the scaled latent the sampler
        # works in, and several formats scale per channel rather than by one
        # factor, so the round trip through model space is not optional here.
        corrected = LatentCorrector(
            colour_space, glue.to_model_space(model, ref), config, mask
        )(glue.to_model_space(model, source))
        out["samples"] = glue.from_model_space(model, corrected)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ApplyLatentColorGuidance": ApplyLatentColorGuidance,
    "LatentColorMatch": LatentColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyLatentColorGuidance": "Apply Latent Color Guidance",
    "LatentColorMatch": "Latent Color Match",
}
