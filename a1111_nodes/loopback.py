"""Automatic1111's Loopback script, rebuilt as a ComfyUI node.

A1111 has no colour correction inside its sampler — reading
``sd_samplers_cfg_denoiser.py``, ``sd_samplers_kdiffusion.py`` and
``sd_samplers_common.py`` turns up no correction on any step; the only
per-step operation touching the latent is the inpaint mask blend. What people
remember as "colour correction carried through the whole process" is
``scripts/loopback.py``: a chain of img2img passes where the output of each
becomes the input of the next, and every pass is corrected against **one**
reference computed once from the original image. The anchor never moves, so
colour cannot drift however many passes run — which from the outside looks
like the correction is built into generation.

That shape cannot be a model patch: each iteration has to decode to pixels,
correct there, and encode again. So this is a sampler node that owns its loop.

Reproduced from the documented behaviour of the script, not copied from it —
that file is AGPL-3.0 and this pack is MIT. The behaviour being reproduced:

* the reference is measured once, from the original image, and reused for
  every iteration;
* iteration 0 runs at the initial denoise, and iteration k at
  ``initial + (final - initial) * curve(k / (loops - 1))``;
* the seed advances by one per iteration;
* each iteration's corrected output is the next iteration's input.

Unlike the latent-space nodes, this one is model-agnostic for free: it works
on decoded pixels, so the latent's channel count never enters into it.
"""

import logging
import math

import numpy as np
import torch

import comfy.samplers
import comfy.utils
import nodes

from ..a1111_cc import color as C

log = logging.getLogger(__name__)

CURVES = ["Linear", "Aggressive", "Lazy"]


def denoise_for_iteration(loop: int, loops: int, initial: float, final: float,
                          curve: str) -> float:
    """The denoise A1111's loopback would use on iteration ``loop``.

    ``loop`` is 1-based here, matching the script: iteration 0 always runs at
    ``initial``, and the curve only starts applying from the second pass.
    """
    if loops <= 1:
        return initial

    progress = loop / (loops - 1)
    if curve == "Aggressive":
        shaped = math.sin(progress * math.pi / 2)
    elif curve == "Lazy":
        shaped = 1.0 - math.cos(progress * math.pi / 2)
    else:
        shaped = progress

    return float(np.clip(initial + (final - initial) * shaped, 0.0, 1.0))


class A1111Loopback:
    """Chain img2img passes, holding every one of them to the first image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE", {"tooltip": "The starting image. Its colours become the "
                                               "anchor every pass is corrected against."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "Advances by one on each pass, as the script does."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "loops": ("INT", {
                    "default": 4, "min": 1, "max": 64,
                    "tooltip": "How many img2img passes to chain.",
                }),
                "denoise": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise for the first pass.",
                }),
                "final_denoise": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise for the last pass. Later passes usually want less, "
                               "so the chain settles instead of wandering.",
                }),
                "denoise_curve": (CURVES, {
                    "default": "Linear",
                    "tooltip": "How denoise travels from the first value to the last.\n"
                               "Linear: evenly.\n"
                               "Aggressive: most of the drop early.\n"
                               "Lazy: most of the drop late.",
                }),
                "color_correction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Correct every pass against the original image — the whole "
                               "point of the chain. Turn it off to see the drift it prevents.",
                }),
                "hold_exposure": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also pin lightness to the original. A1111 never did this — "
                               "its blend takes lightness from the generated image, so a long "
                               "chain can still wander brighter or darker while its colour "
                               "stays put. Switch on when the chain drifts in exposure.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Restrict sampling to this area, as inpainting."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("all_passes", "final")
    FUNCTION = "run"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Automatic1111's Loopback: a chain of img2img passes, each corrected back to the "
        "colours of the original image, so the palette holds however long the chain runs."
    )

    def run(self, model, vae, positive, negative, image, seed, steps, cfg,
            sampler_name, scheduler, loops, denoise, final_denoise,
            denoise_curve, color_correction, hold_exposure=False, mask=None):

        # Measured once, from the original image, and never recomputed — this
        # is what stops the colour walking away over a long chain.
        reference_u8 = None
        if color_correction:
            reference_u8 = self._to_u8(image)

        current = image
        current_seed = int(seed)
        current_denoise = float(denoise)
        passes = []

        progress = comfy.utils.ProgressBar(loops)

        for index in range(loops):
            latent = {"samples": vae.encode(current[:, :, :, :3])}
            if mask is not None:
                latent["noise_mask"] = mask

            sampled, = nodes.common_ksampler(
                model, current_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent, denoise=current_denoise,
            )

            decoded = vae.decode(sampled["samples"])
            if decoded.ndim == 5:        # video models hand back frames
                decoded = decoded.reshape(-1, *decoded.shape[-3:])

            if color_correction:
                decoded = self._correct(decoded, reference_u8, hold_exposure)

            log.info("A1111 Loopback: pass %d/%d at denoise %.3f, seed %d",
                     index + 1, loops, current_denoise, current_seed)

            passes.append(decoded)
            current = decoded
            current_seed += 1
            current_denoise = denoise_for_iteration(
                index + 1, loops, float(denoise), float(final_denoise), denoise_curve)
            progress.update(1)

        # Every pass comes out as a batch so the progression is visible, with
        # the last one also given on its own for wiring onward.
        all_passes = torch.cat(passes, dim=0) if len(passes) > 1 else passes[0]
        return (all_passes, passes[-1])

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _to_u8(image: torch.Tensor) -> np.ndarray:
        x = image.detach().cpu().float().clamp(0.0, 1.0)[..., :3]
        return np.rint(x.numpy() * 255.0).astype(np.uint8)

    @staticmethod
    def _correct(decoded: torch.Tensor, reference_u8: np.ndarray,
                 hold_exposure: bool = False) -> torch.Tensor:
        """Apply the A1111 correction to each frame against the fixed reference."""
        source = A1111Loopback._to_u8(decoded)
        out = np.empty_like(source)
        for i in range(source.shape[0]):
            reference = reference_u8[i] if reference_u8.shape[0] == source.shape[0] \
                else reference_u8[0]
            src = source[i]
            matched = C.match_histograms_u8(C.rgb_to_lab(src), C.rgb_to_lab(reference))
            if hold_exposure:
                # The matched L already sits on the reference's distribution.
                # Keeping it, rather than handing lightness back to the
                # generated frame, is what pins exposure as well as colour.
                out[i] = C.lab_to_rgb(matched)
            else:
                out[i] = C.a1111_luminosity_blend(C.lab_to_rgb(matched), src)

        result = torch.from_numpy(out.astype(np.float32) / 255.0)
        return result.to(device=decoded.device, dtype=decoded.dtype)


NODE_CLASS_MAPPINGS = {
    "A1111Loopback": A1111Loopback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A1111Loopback": "A1111 Loopback (Color Corrected)",
}
