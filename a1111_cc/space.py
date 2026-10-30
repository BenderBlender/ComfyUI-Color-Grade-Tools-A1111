"""The working space a colour correction is computed in.

Raw latent channels do not mean the same thing across model families. SD1.5
and SDXL have four channels that behave roughly like colour axes, so matching
them per channel works. SD3, Flux, Wan and HunyuanVideo have sixteen or more,
where only a few directions carry colour and the rest carry structure and
texture — rescaling all of them changes how much detail the image has, which
is not what anyone asks a colour node for. Flux2 and LTXV go further and
patchify, so a "channel" is partly a position inside a 2x2 block.

The way out is a space every family shares. ComfyUI ships ``latent_rgb_factors``
for 42 of its 45 latent formats: a ``[C,3]`` matrix that approximates the VAE
decoder linearly, used to draw sampling previews. Projecting into it gives one
algorithm for every model::

    rgb   = P^T l + b                      (what the previewer computes)
    rgb'  = f(rgb)                         (the actual colour transfer)
    dl    = P (P^T P)^-1 (rgb' - rgb)      (minimum-norm step back)

The pseudo-inverse matters: the system is three equations in C unknowns, and
the minimum-norm solution puts the whole correction inside ``span(P)`` — the
three directions that carry colour — leaving the other C-3 directions, which
hold structure and texture, untouched by construction. The bias cancels in the
difference.

Where this is approximate, honestly: ``P`` is a least-squares fit to a
non-linear decoder (which is why previews look washed out), so the realised
strength differs from the computed one; and colour information outside
``span(P)`` cannot be reached at all. Both are survivable because the
correction is re-derived from a fresh prediction at every sampling step.
"""

import logging
from typing import Callable, Optional

import torch

log = logging.getLogger(__name__)

RAW = "raw"
RGB_PROJ = "rgb_proj"
PIXEL = "pixel"


def _flux2_inverse_reshape(t: torch.Tensor) -> torch.Tensor:
    """Undo Flux2's un-patchify.

    ``latent_formats.Flux2`` sets::

        reshape: [B,128,H,W] -> [B,32,2,2,H,W] -> permute(0,1,4,2,5,3) -> [B,32,2H,2W]

    so the inverse splits 2H,2W back out and permutes in the opposite order.
    ``tests/test_space.py`` checks the round trip rather than trusting this.
    """
    b, c, h2, w2 = t.shape
    h, w = h2 // 2, w2 // 2
    return (
        t.reshape(b, c, h, 2, w, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * 4, h, w)
    )


# Reshapes we know how to invert, keyed by the latent format's class name.
INVERSE_RESHAPES = {
    "Flux2": _flux2_inverse_reshape,
}


class LatentColorSpace:
    """Maps a latent into the space a correction is computed in, and back."""

    def __init__(self, kind: str, channels: int, ndim: int,
                 proj: Optional[torch.Tensor] = None,
                 bias: Optional[torch.Tensor] = None,
                 reshape: Optional[Callable] = None,
                 inverse_reshape: Optional[Callable] = None,
                 default_method: str = "reinhard",
                 note: str = ""):
        self.kind = kind
        self.channels = channels
        self.ndim = ndim
        self.reshape = reshape
        self.inverse_reshape = inverse_reshape
        self.default_method = default_method
        self.note = note

        self._proj = proj                 # [C, 3]
        self._bias = bias                 # [3]
        self._back = None                 # [3, C], cached
        if proj is not None:
            # back = ((P (P^T P)^-1))^T = (P^T P)^-1 P^T, applied on the last axis
            p = proj.double()
            gram = p.transpose(0, 1) @ p
            self._back = torch.linalg.solve(gram, p.transpose(0, 1)).to(proj.dtype)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def raw(cls, channels: int, ndim: int, default_method: str = "reinhard",
            note: str = "") -> "LatentColorSpace":
        return cls(RAW, channels, ndim, default_method=default_method, note=note)

    @classmethod
    def rgb(cls, factors, bias=None, reshape=None, inverse_reshape=None,
            ndim: int = 4, channels: int = 0) -> "LatentColorSpace":
        p = torch.as_tensor(factors, dtype=torch.float32)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"latent_rgb_factors should be [C,3], got {tuple(p.shape)}")
        b = None if bias is None else torch.as_tensor(bias, dtype=torch.float32)
        return cls(RGB_PROJ, channels or p.shape[0], ndim, proj=p, bias=b,
                   reshape=reshape, inverse_reshape=inverse_reshape)

    @classmethod
    def pixel(cls, ndim: int = 4) -> "LatentColorSpace":
        return cls(PIXEL, 3, ndim, note="latent is already RGB")

    # -- mapping ----------------------------------------------------------- #

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent -> working space."""
        if self.kind in (RAW, PIXEL):
            return latent
        x = self.reshape(latent) if self.reshape is not None else latent
        p = self._proj.to(device=x.device, dtype=x.dtype)
        out = torch.movedim(x, 1, -1) @ p
        if self._bias is not None:
            out = out + self._bias.to(device=x.device, dtype=x.dtype)
        return torch.movedim(out, -1, 1)

    def backward_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """A step in working space -> the matching step in the latent."""
        if self.kind in (RAW, PIXEL):
            return delta
        back = self._back.to(device=delta.device, dtype=delta.dtype)
        out = torch.movedim(delta, 1, -1) @ back
        out = torch.movedim(out, -1, 1)
        if self.inverse_reshape is not None:
            out = self.inverse_reshape(out)
        return out

    def __repr__(self):
        extra = f", {self.note}" if self.note else ""
        return f"<LatentColorSpace {self.kind} C={self.channels} ndim={self.ndim}{extra}>"
