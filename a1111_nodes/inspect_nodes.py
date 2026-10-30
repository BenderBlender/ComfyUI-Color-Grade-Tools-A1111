"""Looking at what the grade did: scopes, and a before/after.

Grading without these is guesswork, and the bigger the wheels the more
confident the guessing gets. A vectorscope says which way a cast runs in a
single glance; a waveform says where the highlights are going; a difference
view says whether a change you can barely see is real.
"""

import logging

import torch

from ..a1111_cc import scopes as S

log = logging.getLogger(__name__)

SCOPE_MODES = ["all", "vectorscope", "waveform", "parade", "histogram"]
COMPARE_MODES = ["side by side", "wipe", "difference", "checkerboard"]


def _to_planes(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE ``[B,H,W,C]`` -> ``[B,3,H,W]``."""
    return image.detach().float().movedim(-1, 1)[:, :3].clamp(0.0, 1.0)


class ColorKitScopes:
    """Vectorscope, waveform and histogram, as a picture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (SCOPE_MODES, {
                    "default": "all",
                    "tooltip": "all: the three side by side.\n"
                               "vectorscope: hue and saturation, brightness discarded. "
                               "The line running up and left is where skin tones sit.\n"
                               "waveform: brightness against horizontal position — a "
                               "column here is a column of the picture.\n"
                               "parade: the same, one panel per channel.\n"
                               "histogram: how many pixels sit at each level.",
                }),
                "size": ("INT", {
                    "default": 512, "min": 192, "max": 1024, "step": 64,
                    "tooltip": "Side of one panel, in pixels.",
                }),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "Look at a latent instead, projected through the "
                               "model's own latent-to-RGB matrix. Needs the model "
                               "connected too. Approximate, like ComfyUI's previews.",
                }),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scopes", "clipping")
    FUNCTION = "measure"
    CATEGORY = "ColorKit/inspect"
    DESCRIPTION = (
        "Vectorscope, waveform and histogram drawn as an image, so a grade can be "
        "judged by what it did rather than by how it looks on an unprofiled monitor."
    )

    def measure(self, image, mode="all", size=512, latent=None, model=None):
        planes = _to_planes(image)

        if latent is not None:
            if model is None:
                log.warning("ColorKit Scopes: a latent needs the model as well, to "
                            "read its latent-to-RGB matrix; showing the image instead.")
            else:
                planes = self._project(latent, model, planes)

        picture = S.compose(planes, mode=mode, size=int(size))

        low, high = S.clipping(planes)
        names = ("R", "G", "B")
        report = "crushed  " + "  ".join(f"{n} {v * 100:.1f}%" for n, v in zip(names, low))
        report += "\nblown    " + "  ".join(f"{n} {v * 100:.1f}%" for n, v in zip(names, high))

        return (picture.unsqueeze(0), report)

    @staticmethod
    def _project(latent, model, fallback):
        from ..a1111_cc import glue

        try:
            samples = latent["samples"]
            space = glue.detect_space(model, samples, "auto")
            if space.kind == "raw":
                log.warning("ColorKit Scopes: this model ships no latent-to-RGB "
                            "matrix, so a latent cannot be shown; using the image.")
                return fallback
            projected = space.forward(glue.to_model_space(model, samples).float())
            if projected.ndim == 5:
                projected = projected[:, :, 0]
            return ((projected + 1.0) * 0.5).clamp(0.0, 1.0)
        except Exception as exc:  # noqa: BLE001 - a scope must not stop a render
            log.warning("ColorKit Scopes: could not read the latent (%r); "
                        "showing the image.", exc)
            return fallback


class ColorKitCompare:
    """Two pictures, one frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Usually the one before."}),
                "image_b": ("IMAGE", {"tooltip": "Usually the one after."}),
                "mode": (COMPARE_MODES, {
                    "default": "wipe",
                    "tooltip": "wipe: B on the right of the line, A on the left.\n"
                               "side by side: both in full.\n"
                               "difference: what changed, amplified — the only honest "
                               "way to see a small grade.\n"
                               "checkerboard: alternating tiles, which makes a seam "
                               "visible wherever the two disagree.",
                }),
                "position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where the wipe sits, or the tile size for the "
                               "checkerboard.",
                }),
                "amplify": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 32.0, "step": 0.5,
                    "tooltip": "How much to multiply the difference by.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "ColorKit/inspect"
    DESCRIPTION = (
        "Put two versions in one frame — wipe, side by side, or an amplified "
        "difference — so a change can be judged rather than remembered."
    )

    def compare(self, image_a, image_b, mode="wipe", position=0.5, amplify=4.0):
        a = image_a.detach().float()
        b = image_b.detach().float()

        if a.shape[1:3] != b.shape[1:3]:
            b = torch.nn.functional.interpolate(
                b.movedim(-1, 1), size=a.shape[1:3], mode="bilinear",
                align_corners=False).movedim(1, -1)
        if b.shape[0] != a.shape[0]:
            b = b[:1].expand(a.shape[0], -1, -1, -1)

        height, width = a.shape[1], a.shape[2]

        if mode == "side by side":
            gap = torch.full((a.shape[0], height, 4, a.shape[-1]), 0.08,
                             dtype=a.dtype, device=a.device)
            out = torch.cat([a, gap, b], dim=2)

        elif mode == "difference":
            out = ((b - a).abs() * amplify).clamp(0.0, 1.0)

        elif mode == "checkerboard":
            tile = max(8, int(min(height, width) * position * 0.25) or 8)
            ys = (torch.arange(height, device=a.device) // tile) % 2
            xs = (torch.arange(width, device=a.device) // tile) % 2
            mask = (ys.reshape(-1, 1) ^ xs.reshape(1, -1)).to(a.dtype)
            out = a * (1 - mask).reshape(1, height, width, 1) + \
                b * mask.reshape(1, height, width, 1)

        else:   # wipe
            split = int(width * position)
            out = a.clone()
            out[:, :, split:] = b[:, :, split:]
            if 0 < split < width:
                edge = min(width - 1, split)
                out[:, :, edge:edge + 2] = 1.0

        return (out.to(device=image_a.device, dtype=image_a.dtype),)


NODE_CLASS_MAPPINGS = {
    "ColorKitScopes": ColorKitScopes,
    "ColorKitCompare": ColorKitCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorKitScopes": "ColorKit Scopes (vectorscope, waveform, histogram)",
    "ColorKitCompare": "ColorKit Compare (before / after)",
}
