"""Turning a reference latent into a correction applied to each x0 prediction."""

import logging
import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch

from . import stats as S
from .space import PIXEL, RAW, LatentColorSpace

log = logging.getLogger(__name__)

# Below this many latent elements per channel the statistics are too noisy to
# act on — small detailer crops land here.
MIN_ELEMENTS = 64


@dataclass
class GuidanceConfig:
    method: str = "reinhard"
    strength: float = 0.5
    max_gain: float = 2.0
    lowpass: int = 0
    temporal: str = "global"
    mkl_shrink: float = 0.1


class LatentCorrector:
    """Holds the reference statistics and applies them to an x0 prediction.

    The reference is measured once, in the working space, at construction
    time; each call only measures the incoming prediction. Everything is done
    in float32 and handed back in the prediction's own dtype, because std over
    fp16 is noticeably noisy.
    """

    def __init__(self, space: LatentColorSpace, reference: torch.Tensor,
                 config: GuidanceConfig, mask: Optional[torch.Tensor] = None):
        self.space = space
        self.cfg = config
        self.mask = mask
        self._warned = set()
        self._mask_key = None
        self._mask_value = None

        if config.method not in S.MATCHERS:
            raise ValueError(f"unknown method {config.method!r}; "
                             f"expected one of {sorted(S.MATCHERS)}")

        if config.method == "mkl" and config.temporal != "global":
            log.warning(
                "latent colour guidance: MKL needs one covariance for the whole "
                "clip, which per-frame reduction cannot give; using reinhard."
            )
            # replace rather than assign: the caller's config is theirs
            config = dataclasses.replace(config, method="reinhard")
        self.cfg = config

        ref_ws = space.forward(reference.float())
        dims = S.reduce_dims(ref_ws, config.temporal)
        # The reference is summarised globally even in per_frame mode: a
        # reference frame count rarely matches the clip's, and a single target
        # is what keeps a video from drifting.
        if ref_ws.ndim == 5:
            dims = [2, 3, 4]
        self.ref_stats = S.channel_stats(
            ref_ws, dims, need_cov=(config.method == "mkl")
        )

    # -- helpers ----------------------------------------------------------- #

    def warn_once(self, key: str, msg: str, *args):
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _mask_for(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        """Fit the mask to whatever is actually being sampled.

        The reference is allowed to be a different size from the latent under
        the sampler — statistics do not care — and detailers hand over small
        crops, so the mask has to be resized per call, not once up front.
        """
        if self.mask is None:
            return None
        # One slot, not a dict: a detailer works through one crop size at a
        # time, so this hits just as often and cannot grow without bound.
        key = (tuple(like.shape), like.device, like.dtype)
        if self._mask_key == key:
            return self._mask_value

        m = self.mask.float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        target = like.shape[-2:]
        flat = m.reshape(-1, 1, *m.shape[-2:])
        if tuple(flat.shape[-2:]) != tuple(target):
            flat = torch.nn.functional.interpolate(
                flat, size=tuple(target), mode="bilinear", align_corners=False)
        m = flat.reshape(*m.shape[:-2], *target)

        if m.shape[0] not in (1, like.shape[0]):
            m = m[:1]
        m = m.to(device=like.device, dtype=like.dtype)
        self._mask_key, self._mask_value = key, m
        return m

    def _ref_for(self, batch: int) -> S.ChannelStats:
        rb = self.ref_stats.mean.shape[0]
        if rb in (1, batch):
            return self.ref_stats
        raise ValueError(
            f"the reference latent has a batch of {rb}, but sampling a batch of "
            f"{batch}; give the reference one frame or exactly {batch}"
        )

    # -- the correction ---------------------------------------------------- #

    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        if x0.ndim not in (4, 5):
            self.warn_once("ndim", "latent colour guidance: skipping a %dD prediction", x0.ndim)
            return x0

        # Checked for every space, not just the raw one: a projection with the
        # wrong channel count fails inside the matmul, mid-sample.
        if x0.shape[1] != self.space.channels:
            self.warn_once(
                "channels",
                "latent colour guidance: prediction has %d channels but the reference "
                "has %d — skipping. Encode the reference with this model's own VAE.",
                x0.shape[1], self.space.channels,
            )
            return x0

        src = x0.float()
        ws = self.space.forward(src)

        dims = S.reduce_dims(ws, self.cfg.temporal)
        elements = 1
        for d in dims:
            elements *= ws.shape[d]
        if elements < MIN_ELEMENTS:
            self.warn_once(
                "small",
                "latent colour guidance: only %d latent elements per channel — "
                "too few to measure, skipping (small detailer crop?)", elements,
            )
            return x0

        try:
            ref_stats = self._ref_for(ws.shape[0])
        except ValueError as exc:
            self.warn_once("batch", "latent colour guidance: %s", exc)
            return x0

        need_cov = self.cfg.method == "mkl"
        src_stats = S.channel_stats(ws, dims, mask=self._mask_for(ws), need_cov=need_cov)
        if src_stats.count < MIN_ELEMENTS:
            # A mask that selects almost nothing gives a near-zero spread, and
            # an unguarded gain would then rescale the entire latent.
            self.warn_once(
                "mask_empty",
                "latent colour guidance: the mask covers only ~%.0f latent "
                "elements — too few to measure, skipping.", src_stats.count,
            )
            return x0
        ref_stats = ref_stats.to(ws.device, ws.dtype)

        matcher = S.MATCHERS[self.cfg.method]
        if self.cfg.method == "mkl":
            corrected = matcher(ws, src_stats, ref_stats,
                                max_gain=self.cfg.max_gain, shrink=self.cfg.mkl_shrink)
        elif self.cfg.method == "reinhard":
            corrected = matcher(ws, src_stats, ref_stats, max_gain=self.cfg.max_gain)
        else:
            corrected = matcher(ws, src_stats, ref_stats)

        delta_ws = corrected - ws
        if self.cfg.lowpass:
            delta_ws = S.lowpass(delta_ws, self.cfg.lowpass)

        delta = self.space.backward_delta(delta_ws)
        if delta.shape != src.shape:
            self.warn_once(
                "shape",
                "latent colour guidance: correction came back as %s for a %s "
                "prediction — skipping", tuple(delta.shape), tuple(src.shape),
            )
            return x0

        return (src + self.cfg.strength * delta).to(x0.dtype)


def correct_latent(space: LatentColorSpace, source: torch.Tensor,
                   reference: torch.Tensor, config: GuidanceConfig,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One-shot version of the above, for the static LATENT -> LATENT node."""
    return LatentCorrector(space, reference, config, mask)(source)
