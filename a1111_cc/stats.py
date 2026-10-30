"""Channel statistics and the linear colour transfers built on them.

Pure torch — nothing here imports ComfyUI, so it can be tested against
synthetic tensors. Shapes are ``[B, C, H, W]`` for image latents and
``[B, C, T, H, W]`` for video ones; the channel axis is always 1.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

EPS = 1e-6


def reduce_dims(x: torch.Tensor, temporal: str = "global") -> List[int]:
    """Which axes the statistics are gathered over.

    4D latents always reduce over H,W. For 5D (video) latents, ``global``
    also folds in T — one correction for the whole clip, which cannot
    flicker — while ``per_frame`` keeps T separate.
    """
    if x.ndim == 4:
        return [2, 3]
    if x.ndim == 5:
        return [2, 3, 4] if temporal == "global" else [3, 4]
    raise ValueError(f"expected a 4D or 5D latent, got {x.ndim}D")


@dataclass
class ChannelStats:
    """Per-channel mean/std (and optionally covariance) of a latent."""

    mean: torch.Tensor            # broadcastable against the source
    std: Optional[torch.Tensor]
    cov: Optional[torch.Tensor]   # [B, C, C]
    count: float                  # effective number of elements per channel

    def to(self, device, dtype):
        return ChannelStats(
            self.mean.to(device=device, dtype=dtype),
            None if self.std is None else self.std.to(device=device, dtype=dtype),
            None if self.cov is None else self.cov.to(device=device, dtype=dtype),
            self.count,
        )


def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Line a mask up with a ``[B,C,(T,)H,W]`` latent.

    Accepts ``[B,H,W]`` and ``[B,T,H,W]`` for video latents; a flat mask with
    no time axis is applied to every frame, which is what a user who drew one
    mask for a clip means.
    """
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)                       # [H,W] -> [1,H,W]
    if m.ndim == like.ndim - 1:
        m = m.unsqueeze(1)                       # add the channel axis
    elif like.ndim == 5 and m.ndim == like.ndim - 2:
        # Only for video: a flat [B,H,W] mask applies to every frame. Allowing
        # this for 4D as well would turn [H,W] into [H,1,1,W] and weight the
        # batch axis by height — wrong, and silently so when H happens to be B.
        m = m.unsqueeze(1).unsqueeze(1)
    if m.ndim != like.ndim:
        raise ValueError(f"mask of {mask.ndim}D cannot address a {like.ndim}D latent")
    if m.shape[0] != like.shape[0] and m.shape[0] == 1:
        m = m.expand(like.shape[0], *m.shape[1:])
    return m.to(device=like.device, dtype=like.dtype)


def channel_stats(x: torch.Tensor, dims: List[int], mask: Optional[torch.Tensor] = None,
                  need_cov: bool = False, eps: float = EPS) -> ChannelStats:
    """Mean/std (and covariance) per channel, optionally weighted by a mask."""
    x = x.float()

    if mask is None:
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        count = 1.0
        for d in dims:
            count *= x.shape[d]
    else:
        w = _broadcast_mask(mask, x).float()
        total = w.sum(dim=dims, keepdim=True).clamp_min(eps)
        mean = (x * w).sum(dim=dims, keepdim=True) / total
        var = (w * (x - mean) ** 2).sum(dim=dims, keepdim=True) / total
        std = var.clamp_min(0.0).sqrt()
        # Kish's effective sample size, so a faint mask over the whole frame
        # counts as the few pixels it is really worth rather than all of them.
        squared = (w ** 2).sum(dim=dims, keepdim=True).clamp_min(eps)
        # Median rather than minimum: with per-frame reduction a single empty
        # frame would otherwise speak for the whole clip and disable it.
        count = float((total ** 2 / squared).median())

    cov = None
    if need_cov:
        # Covariance is only defined here when the statistics are per [B,C]:
        # with per-frame reduction the mean carries a time axis and there is
        # no single matrix to build. The nodes steer MKL away from that case.
        if mean.shape[2:] != torch.Size([1] * (x.ndim - 2)):
            raise ValueError(
                "covariance needs statistics reduced over every spatial axis; "
                "per-frame reduction leaves a time axis in place"
            )
        b, c = x.shape[0], x.shape[1]
        flat = x.reshape(b, c, -1)
        mu = mean.reshape(b, c, 1)
        if mask is None:
            centred = flat - mu
            n = flat.shape[-1]
            cov = centred @ centred.transpose(1, 2) / max(n - 1, 1)
        else:
            w = _broadcast_mask(mask, x).float().reshape(b, -1, flat.shape[-1])
            w = w[:, :1, :]  # the mask is shared across channels
            centred = (flat - mu) * w.sqrt()
            cov = centred @ centred.transpose(1, 2) / w.sum(dim=-1).clamp_min(eps).unsqueeze(-1)

    return ChannelStats(mean=mean, std=std, cov=cov, count=count)


# --------------------------------------------------------------------------- #
# transfers
# --------------------------------------------------------------------------- #

def _gain(src_std: torch.Tensor, ref_std: torch.Tensor, max_gain: float,
          eps: float = EPS) -> torch.Tensor:
    """Per-channel scale, clamped so a nearly-flat source cannot explode.

    Early in sampling the x0 prediction is almost featureless, so ``src_std``
    approaches zero and an unclamped ratio would amplify whatever noise is
    there into visible artefacts.
    """
    g = ref_std / (src_std + eps)
    if max_gain and max_gain > 0:
        g = g.clamp(1.0 / max_gain, max_gain)
    return g


def match_mean_only(src: torch.Tensor, src_st: ChannelStats,
                    ref_st: ChannelStats, **_) -> torch.Tensor:
    """Shift each channel onto the reference's mean, leaving spread untouched.

    The safest transfer there is: it can never change contrast or the amount
    of detail, which matters for the many-channel latents where most channels
    are structure rather than colour.
    """
    return src + (ref_st.mean - src_st.mean)


def match_reinhard(src: torch.Tensor, src_st: ChannelStats, ref_st: ChannelStats,
                   max_gain: float = 2.0, match_mean: bool = True,
                   match_std: bool = True, eps: float = EPS) -> torch.Tensor:
    """Classic per-channel mean/std transfer (Reinhard et al.)."""
    out = src - src_st.mean
    if match_std and src_st.std is not None and ref_st.std is not None:
        out = out * _gain(src_st.std, ref_st.std, max_gain, eps)
    out = out + (ref_st.mean if match_mean else src_st.mean)
    return out


def _sym_sqrt(mat: torch.Tensor, inverse: bool = False,
              eps: float = 1e-8) -> torch.Tensor:
    """Square root (or inverse square root) of a symmetric PSD matrix."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(eps)
    root = evals.rsqrt() if inverse else evals.sqrt()
    return evecs @ torch.diag_embed(root) @ evecs.transpose(-1, -2)


def match_mkl(src: torch.Tensor, src_st: ChannelStats, ref_st: ChannelStats,
              max_gain: float = 2.0, shrink: float = 0.1,
              eps: float = EPS) -> torch.Tensor:
    """Monge-Kantorovich linear transfer (Pitie & Kokaram).

    The unique symmetric positive-definite map taking ``N(mu_s, S_s)`` to
    ``N(mu_r, S_r)`` at least transport cost::

        T = S_s^-1/2 (S_s^1/2 S_r S_s^1/2)^1/2 S_s^-1/2

    Unlike :func:`match_reinhard` it accounts for correlations between
    channels, so a cast that lies diagonally across them is actually removed
    rather than approximated axis by axis.

    Covariances are shrunk towards a scaled identity first, because an x0
    prediction early in sampling is close to rank-deficient and the raw
    inverse square root would blow up.
    """
    if src_st.cov is None or ref_st.cov is None:
        raise ValueError("MKL needs covariances; call channel_stats(need_cov=True)")

    b, c = src.shape[0], src.shape[1]

    def _shrink(cov):
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
        eye = torch.eye(c, device=cov.device, dtype=cov.dtype).expand_as(cov)
        return (1.0 - shrink) * cov + shrink * (trace / c).view(-1, 1, 1) * eye

    cs = _shrink(src_st.cov.float())
    cr = _shrink(ref_st.cov.float())
    if cr.shape[0] == 1 and b > 1:
        cr = cr.expand(b, c, c)

    cs_half = _sym_sqrt(cs)
    cs_inv_half = _sym_sqrt(cs, inverse=True)
    middle = _sym_sqrt(cs_half @ cr @ cs_half)
    t = cs_inv_half @ middle @ cs_inv_half

    if max_gain and max_gain > 0:
        evals, evecs = torch.linalg.eigh(0.5 * (t + t.transpose(-1, -2)))
        evals = evals.clamp(1.0 / max_gain, max_gain)
        t = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)

    flat = (src - src_st.mean).reshape(b, c, -1)
    out = (t @ flat).reshape(src.shape)
    return out + ref_st.mean


MATCHERS = {
    "mean_only": match_mean_only,
    "reinhard": match_reinhard,
    "mkl": match_mkl,
}


# --------------------------------------------------------------------------- #
# modifiers
# --------------------------------------------------------------------------- #

def lowpass(delta: torch.Tensor, radius: int) -> torch.Tensor:
    """Blur a correction over H,W so only its broad part survives.

    With a per-channel gain above 1 the transfer also amplifies fine detail;
    smoothing the delta keeps the colour shift and drops that side effect.
    T is left alone — smearing a correction across time is what causes ghosting.
    """
    if radius <= 0:
        return delta
    k = 2 * radius + 1
    shape = delta.shape
    if delta.ndim == 5:
        # fold T into the batch so avg_pool2d only ever sees H,W
        flat = delta.permute(0, 2, 1, 3, 4).reshape(shape[0] * shape[2], shape[1], *shape[3:])
    else:
        flat = delta

    pad = radius
    padded = torch.nn.functional.pad(flat, (pad, pad, pad, pad), mode="replicate")
    smoothed = torch.nn.functional.avg_pool2d(padded, kernel_size=k, stride=1)

    if delta.ndim == 5:
        smoothed = smoothed.reshape(shape[0], shape[2], shape[1], *shape[3:]).permute(0, 2, 1, 3, 4)
    return smoothed


def blend(src: torch.Tensor, corrected: torch.Tensor, strength: float) -> torch.Tensor:
    """``src + strength * (corrected - src)``, in the source's dtype."""
    if strength == 1.0:
        return corrected.to(src.dtype)
    return (src.float() + strength * (corrected.float() - src.float())).to(src.dtype)
