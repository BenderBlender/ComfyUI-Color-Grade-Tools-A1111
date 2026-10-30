"""Automatic1111's Loopback script, rebuilt as a ComfyUI node.

A1111 has no colour correction inside its sampler — reading
``sd_samplers_cfg_denoiser.py``, ``sd_samplers_kdiffusion.py`` and
``sd_samplers_common.py`` turns up no correction on any step; the only
per-step operation touching the latent is the inpaint mask blend. What people
remember as "colour correction carried through the whole process" is
``scripts/loopback.py``: a chain of img2img passes where the output of each
becomes the input of the next, and every pass is corrected against **one**
reference measured once from the original image. The anchor never moves, so
colour cannot drift however many passes run, while the content is fully
regenerated each time — which from the outside looks like the correction is
part of generation.

That shape cannot be a model patch: each iteration has to decode to pixels,
correct there, and encode again. So this is a sampler node that owns its loop.

Reproduced from the documented behaviour of the script, not copied from it —
that file is AGPL-3.0 and this pack is MIT. The behaviour reproduced:

* the reference is measured once, from the original image, and reused by every
  pass;
* pass 0 runs at the initial denoise, and pass k at
  ``initial + (final - initial) * curve(k / (loops - 1))``;
* the seed advances by one per pass;
* each pass's corrected output is the next pass's input;
* with a mask, the untouched area is pasted back from the original after each
  pass, which is what ``apply_overlay`` does there — without it that area
  would be re-encoded and re-decoded on every pass and slowly soften.

Unlike the latent-space nodes this one is model-agnostic for free: it works on
decoded pixels, so the latent's channel count never enters into it.
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
SEED_MASK = 0xffffffffffffffff
MAX_LIGHTNESS_GAIN = 2.0


def denoise_for_iteration(loop: int, loops: int, initial: float, final: float,
                          curve: str) -> float:
    """The denoise A1111's loopback uses on pass ``loop`` (0-based).

    Pass 0 always runs at ``initial``; the curve applies from the second pass
    onward, reaching ``final`` exactly on the last one.
    """
    if loops <= 1 or loop <= 0:
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
                                               "anchor every pass is corrected against. "
                                               "Sides are cropped to what the VAE accepts."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MASK,
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
                "hold_exposure": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Also pull lightness back towards the original (0 = off).\n"
                               "A1111 never did this: its blend takes lightness from the "
                               "generated image, so a long chain holds its colour but can "
                               "still wander brighter or darker. This matches the average "
                               "and spread of lightness rather than its whole curve, so "
                               "deliberate changes in lighting survive.",
                }),
                "keep_all_passes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Return every pass as a batch. Turn off for long chains at "
                               "high resolution, where holding them all costs real memory.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Sample only inside this area. Everything outside is pasted "
                               "back from the original after each pass, so it does not "
                               "soften through repeated encoding.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("all_passes", "final")
    FUNCTION = "run"
    CATEGORY = "ColorKit/sampling"
    DESCRIPTION = (
        "Automatic1111's Loopback: a chain of img2img passes, each corrected back to the "
        "colours of the original image, so the palette holds however long the chain runs."
    )

    def run(self, model, vae, positive, negative, image, seed, steps, cfg,
            sampler_name, scheduler, loops, denoise, final_denoise,
            denoise_curve, color_correction, hold_exposure=0.0,
            keep_all_passes=True, mask=None):

        if image.shape[-1] == 1:
            image = image.repeat(1, 1, 1, 3)        # grayscale IMAGE, as the VAE wants 3

        if hold_exposure > 0.0 and not color_correction:
            log.warning("A1111 Loopback: hold_exposure does nothing while "
                        "color_correction is off.")

        # Measured once, from the original, and never recomputed — this is what
        # stops the colour walking away over a long chain. Stored as LAB, which
        # is the form both the match and the lightness pin need.
        reference_lab = None
        if color_correction:
            reference_lab = np.stack([C.rgb_to_lab(frame)
                                      for frame in self._to_u8(image)], axis=0)

        current = image
        original = image
        passes = []
        warned = set()
        progress = comfy.utils.ProgressBar(loops)

        for index in range(loops):
            current_denoise = denoise_for_iteration(
                index, loops, float(denoise), float(final_denoise), denoise_curve)
            current_seed = (int(seed) + index) & SEED_MASK   # never overflow manual_seed

            latent = {"samples": vae.encode(current)}
            if mask is not None:
                latent["noise_mask"] = mask

            if current_denoise <= 0.0:
                # ComfyUI builds an empty sigma schedule at zero denoise and the
                # sampler then indexes into it. A1111 simply runs no steps.
                sampled = latent
            else:
                sampled, = nodes.common_ksampler(
                    model, current_seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=current_denoise,
                )

            decoded = vae.decode(sampled["samples"])
            if decoded.ndim == 5:        # video models hand back frames
                decoded = decoded.reshape(-1, *decoded.shape[-3:])

            if color_correction:
                decoded = self._correct(decoded, reference_lab, hold_exposure, warned)

            if mask is not None:
                decoded = self._paste_back(decoded, original, mask)

            log.info("A1111 Loopback: pass %d/%d, denoise %.3f, seed %d",
                     index + 1, loops, current_denoise, current_seed)

            if keep_all_passes:
                passes.append(decoded)
            current = decoded
            progress.update(1)

        final = current
        if not keep_all_passes or len(passes) == 1:
            return (final, final)

        shapes = {tuple(p.shape[1:]) for p in passes}
        if len(shapes) > 1:
            # A video VAE can hand back a different frame count per pass; those
            # cannot be stacked, and failing at the very end of a long chain
            # would be the worst possible moment.
            log.warning("A1111 Loopback: passes came back at different sizes %s; "
                        "returning the last one only.", sorted(shapes))
            return (final, final)

        return (torch.cat(passes, dim=0), final)

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _to_u8(image: torch.Tensor) -> np.ndarray:
        x = image.detach().cpu().float().clamp(0.0, 1.0)[..., :3]
        return np.rint(x.numpy() * 255.0).astype(np.uint8)

    @staticmethod
    def _pin_lightness(rgb_u8: np.ndarray, reference_lab: np.ndarray,
                       strength: float) -> np.ndarray:
        """Pull lightness towards the reference's average and spread.

        Deliberately not a histogram match: that would impose the original's
        whole tonal curve, flattening any lighting the chain introduced on
        purpose and banding flat areas at 8 bits. Matching the first two
        moments corrects drift and leaves the rest alone.
        """
        lab = C.rgb_to_lab(rgb_u8).astype(np.float32)
        lightness = lab[..., 0]
        ref = reference_lab[..., 0].astype(np.float32)

        src_mean, src_std = lightness.mean(), lightness.std()
        gain = float(np.clip(ref.std() / (src_std + 1e-6),
                             1.0 / MAX_LIGHTNESS_GAIN, MAX_LIGHTNESS_GAIN))
        target = (lightness - src_mean) * gain + ref.mean()

        lab[..., 0] = lightness + strength * (target - lightness)
        return C.lab_to_rgb(np.rint(lab).clip(0, 255).astype(np.uint8))

    @staticmethod
    def _correct(decoded: torch.Tensor, reference_lab: np.ndarray,
                 hold_exposure: float, warned: set) -> torch.Tensor:
        """Apply the A1111 correction to each frame against the fixed reference."""
        source = A1111Loopback._to_u8(decoded)
        frames, ref_frames = source.shape[0], reference_lab.shape[0]
        if ref_frames != frames and ref_frames != 1 and "frames" not in warned:
            warned.add("frames")
            log.warning("A1111 Loopback: %d frame(s) came back against %d reference "
                        "frame(s); using the first reference for all of them.",
                        frames, ref_frames)

        out = np.empty_like(source)
        for i in range(frames):
            ref_lab = reference_lab[i] if ref_frames == frames else reference_lab[0]
            src = source[i]
            matched = C.match_histograms_u8(C.rgb_to_lab(src), ref_lab)
            corrected = C.a1111_luminosity_blend(C.lab_to_rgb(matched), src)
            if hold_exposure > 0.0:
                corrected = A1111Loopback._pin_lightness(corrected, ref_lab, hold_exposure)
            out[i] = corrected

        result = torch.from_numpy(out.astype(np.float32) / 255.0)
        return result.to(device=decoded.device, dtype=decoded.dtype)

    @staticmethod
    def _paste_back(decoded: torch.Tensor, original: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """Restore the unmasked area from the original, as ``apply_overlay`` does.

        Without this the area the sampler never touched still goes through
        encode and decode on every pass, and a long chain visibly softens it.
        """
        height, width = decoded.shape[1], decoded.shape[2]

        source = original.to(device=decoded.device, dtype=decoded.dtype)
        if source.shape[1:3] != (height, width):
            top = max(0, (source.shape[1] - height) // 2)
            left = max(0, (source.shape[2] - width) // 2)
            source = source[:, top:top + height, left:left + width, :]
        if source.shape[1:3] != (height, width):
            log.warning("A1111 Loopback: cannot line the original up with the result "
                        "(%s against %s); skipping the paste-back.",
                        tuple(source.shape[1:3]), (height, width))
            return decoded
        if source.shape[0] != decoded.shape[0]:
            source = source[:1].expand(decoded.shape[0], -1, -1, -1)

        weights = mask.to(device=decoded.device, dtype=decoded.dtype)
        if weights.ndim == 2:
            weights = weights.unsqueeze(0)
        weights = torch.nn.functional.interpolate(
            weights.unsqueeze(1), size=(height, width),
            mode="bilinear", align_corners=False).squeeze(1)
        if weights.shape[0] != decoded.shape[0]:
            weights = weights[:1].expand(decoded.shape[0], -1, -1)
        weights = weights.clamp(0.0, 1.0).unsqueeze(-1)

        return decoded * weights + source * (1.0 - weights)


NODE_CLASS_MAPPINGS = {
    "A1111Loopback": A1111Loopback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A1111Loopback": "ColorKit A1111 Loopback",
}
