"""A1111-style img2img colour correction, in image space."""

import logging

import numpy as np
import torch

from ..a1111_cc import color as C

log = logging.getLogger(__name__)

BLEND_MODES = ["a1111", "lab_color", "lab_luminosity", "none"]
MATCH_CHANNELS = ["L+a+b", "a+b (chroma only)"]


def _to_u8(image: torch.Tensor):
    """ComfyUI IMAGE ``[B,H,W,C]`` in [0,1] -> ``(uint8 [B,H,W,3], alpha or None)``."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("expected an IMAGE tensor")
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected IMAGE of shape [B,H,W,1/3/4], got {tuple(image.shape)}")

    x = image.detach().cpu().float().clamp(0.0, 1.0)
    alpha = None
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 1, 3)
    elif x.shape[-1] == 4:
        alpha = x[..., 3:]
        x = x[..., :3]
    return np.rint(x.numpy() * 255.0).astype(np.uint8), alpha


def _from_u8(arr: np.ndarray, like: torch.Tensor, alpha=None) -> torch.Tensor:
    out = torch.from_numpy(arr.astype(np.float32) / 255.0)
    if alpha is not None:
        out = torch.cat([out, alpha], dim=-1)
    return out.to(device=like.device, dtype=like.dtype)


def _pick(batch: np.ndarray, i: int, size: int, what: str) -> np.ndarray:
    """Index into a batch that is either full length or a single shared frame."""
    if batch.shape[0] == size:
        return batch[i]
    if batch.shape[0] == 1:
        return batch[0]
    raise ValueError(
        f"batch mismatch: image has {size} frame(s) but {what} has {batch.shape[0]}; "
        f"{what} must either match the image batch or hold a single frame"
    )


class A1111ColorCorrection:
    """Pull an image's palette back onto a reference, the way A1111 img2img does."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to correct — usually the generated one."}),
                "reference": ("IMAGE", {"tooltip": "The image whose colours should be adopted, "
                                                   "usually the img2img input."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = image untouched, 1 = fully corrected.",
                }),
                "blend_mode": (BLEND_MODES, {
                    "default": "a1111",
                    "tooltip": "a1111: exactly what the webui does — chroma from the corrected "
                               "image, brightness from the original.\n"
                               "lab_color: the same intent approximated in LAB.\n"
                               "lab_luminosity: the inverse — keeps the original chroma and takes "
                               "the corrected brightness.\n"
                               "none: the histogram-matched result, unblended.",
                }),
                "match_channels": (MATCH_CHANNELS, {
                    "default": "L+a+b",
                    "tooltip": "Which LAB channels the histogram match is allowed to touch. "
                               "Chroma only leaves contrast and exposure alone.",
                }),
                "ref_blur": ("INT", {
                    "default": 0, "min": 0, "max": 201, "step": 1,
                    "tooltip": "Blur the reference first (0 = off), so only its overall cast is "
                               "transferred rather than local detail. Not part of A1111.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Restrict the correction to the white areas, and measure the "
                               "histograms there too.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Color Grade Tools/image"
    DESCRIPTION = (
        "What the Automatic1111 webui does when you tick 'colour correction' on an "
        "img2img pass, reproduced exactly.\n\n"
        "WHAT IT IS FOR. Any img2img or refine pass drifts in colour, and the higher "
        "the denoise the further it goes. Give this the result and the picture you "
        "started from, and it puts the palette back.\n\n"
        "HOW IT WORKS. The result's histogram is matched to the reference in LAB, then "
        "blended so the generated image keeps its own brightness and takes only the "
        "reference's colour. That last part is the whole operation: the default "
        "blend_mode 'a1111' is the webui's behaviour, while 'lab_luminosity' is the "
        "opposite trade and keeps the drift.\n\n"
        "WHERE IT GOES. After VAEDecode. Wire the img2img input picture into "
        "'reference'. Exact and cheap — nothing is sampled."
    )

    def apply(self, image, reference, strength=1.0, blend_mode="a1111",
              match_channels="L+a+b", ref_blur=0, mask=None):
        if strength <= 0.0:
            return (image,)

        src_batch, alpha = _to_u8(image)
        ref_batch, _ = _to_u8(reference)
        size = src_batch.shape[0]

        chroma_only = match_channels.startswith("a+b")
        if chroma_only and blend_mode in ("a1111", "lab_color"):
            log.warning(
                "A1111ColorCorrection: 'match_channels=%s' has no effect with "
                "blend_mode='%s', because that blend already discards the corrected "
                "brightness. Use blend_mode='none' to see it.",
                match_channels, blend_mode,
            )

        mask_batch = None
        if mask is not None:
            m = mask.detach().cpu().float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            mask_batch = m.clamp(0.0, 1.0).numpy()

        out_batch = np.empty_like(src_batch)
        for i in range(size):
            src = src_batch[i]
            ref = _pick(ref_batch, i, size, "reference")

            if ref_blur > 0:
                ref = C.gaussian_blur_u8(ref, ref_blur)

            weight = float(strength)
            sel = None
            if mask_batch is not None:
                m = _pick(mask_batch, i, size, "mask")
                m = C.resize_bilinear(m, src.shape[1], src.shape[0])
                weight = (m * float(strength))[..., None]
                sel = m > 0.5

            src_lab = C.rgb_to_lab(src)
            ref_lab = C.rgb_to_lab(ref)
            matched_lab = C.match_histograms_masked_u8(src_lab, ref_lab, src_mask=sel)
            if chroma_only:
                matched_lab[..., 0] = src_lab[..., 0]
            corrected = C.lab_to_rgb(matched_lab)

            if blend_mode == "a1111":
                corrected = C.a1111_luminosity_blend(corrected, src)
            elif blend_mode == "lab_color":
                corrected = C.lab_color_blend(corrected, src)
            elif blend_mode == "lab_luminosity":
                corrected = C.lab_luminosity_blend(corrected, src)

            out_batch[i] = C.mix_u8(src, corrected, weight)

        return (_from_u8(out_batch, image, alpha),)


NODE_CLASS_MAPPINGS = {
    "A1111ColorCorrection": A1111ColorCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A1111ColorCorrection": "Color Grade · A1111 Color Correction",
}
