"""The original KSampler node, kept so existing workflows still open.

It is a thin shell now: it patches the model with the same guidance the new
node applies and hands everything else to ComfyUI's own ``common_ksampler``,
which is what the first version had copied — and had drifted from, losing
noise masks, batch indices and latent previews along the way.

Marked ``DEPRECATED`` so it stays out of the node menu while saved graphs keep
loading.
"""

import logging

import comfy.samplers
import nodes

from ..a1111_cc import glue
from ..a1111_cc.guidance import GuidanceConfig, LatentCorrector

log = logging.getLogger(__name__)

_WARNED = False


class A1111LatentColorKSampler:
    """Deprecated. Use Apply Latent Color Guidance with any sampler instead."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        # The order here is load-bearing: saved workflows store widget values
        # positionally, so it has to stay exactly as the first version had it.
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "start_step_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_step_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/advanced"
    DESCRIPTION = (
        "Deprecated. Replace with Apply Latent Color Guidance feeding an ordinary "
        "KSampler — the same correction, with every sampler and model family."
    )

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise,
               reference_latent, strength, start_step_ratio, end_step_ratio):
        global _WARNED
        if not _WARNED:
            _WARNED = True
            log.warning(
                "KSampler (A1111 Latent Color Match) is deprecated. It now corrects the "
                "x0 prediction with a correctly scaled reference, so the same strength "
                "value is far stronger than before — start around 0.3-0.5 rather than "
                "0.01. The step ratios are now read as a share of the model's sigma "
                "schedule rather than of the step count, which shifts the window a "
                "little on non-linear schedulers such as karras. "
                "Replace it with Apply Latent Color Guidance + KSampler."
            )

        if strength > 0.0 and start_step_ratio < end_step_ratio:
            reference = glue.fit_reference_dims(model, reference_latent["samples"])
            colour_space = glue.detect_space(model, reference, "auto")
            config = GuidanceConfig(
                method=colour_space.default_method,
                strength=float(strength),
            )
            corrector = LatentCorrector(
                colour_space, glue.to_model_space(model, reference), config
            )
            sigma_hi, sigma_lo = glue.sigma_window(model, start_step_ratio, end_step_ratio)
            model = glue.attach(model, corrector, sigma_hi, sigma_lo)

        return nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=denoise,
        )


NODE_CLASS_MAPPINGS = {
    "A1111LatentColorKSampler": A1111LatentColorKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A1111LatentColorKSampler": "KSampler (A1111 Latent Color Match, deprecated)",
}
