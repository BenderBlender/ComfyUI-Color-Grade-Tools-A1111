"""The only module that knows about ComfyUI internals.

Keeping the imports here means everything in :mod:`a1111_cc.stats`,
:mod:`a1111_cc.space` and :mod:`a1111_cc.guidance` can be tested with plain
torch tensors and no ComfyUI on the path.
"""

import logging

import torch

from . import axes
from .space import INVERSE_RESHAPES, LatentColorSpace

log = logging.getLogger(__name__)

HOOK_KEY = "a1111_latent_color_guidance"


def to_model_space(model, samples: torch.Tensor) -> torch.Tensor:
    """Scale a LATENT the way the sampler does before it starts.

    ``CFGGuider.inner_sample`` runs the latent through ``process_latent_in``,
    so everything a sampling hook sees is in that space, while a LATENT socket
    carries the unscaled VAE output. Skipping this step was the original
    node's worst bug: for SDXL the reference arrived about 7.7x out of scale,
    which is why its strength defaulted to 0.01.
    """
    latent_format = model.get_model_object("latent_format")
    return latent_format.process_in(samples)


def from_model_space(model, samples: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`to_model_space`, for results handed back as LATENT."""
    return model.get_model_object("latent_format").process_out(samples)


def detect_space(model, reference: torch.Tensor, requested: str = "auto",
                 measured_axes: bool = False) -> LatentColorSpace:
    """Work out which space to correct in, from the model's latent format.

    With ``measured_axes`` the decoder's measured colour response (:mod:`axes`)
    replaces the preview matrix when this family has been measured. The grade
    wheels ask for it: they *invert* the matrix to write colour into the
    latent, and the preview fit inverts badly (SDXL: magenta, 3-5x too strong).
    The guidance and match nodes keep the preview matrix by default — they
    match statistics measured through the same matrix, where it is a
    consistent pair, and their results were not measured with the other one.
    """
    latent_format = model.get_model_object("latent_format")
    name = type(latent_format).__name__

    dimensions = getattr(latent_format, "latent_dimensions", 2)
    if dimensions == 1:
        raise ValueError(
            f"{name} is a 1-D latent format (audio, or a 3-D asset format). Its "
            f"channels carry no colour, so colour matching does not apply here."
        )

    ndim = 4 if dimensions == 2 else 5
    channels = reference.shape[1]
    declared = getattr(latent_format, "latent_channels", None)
    if declared is not None and declared != channels:
        raise ValueError(
            f"the reference latent has {channels} channels but {name} expects "
            f"{declared}; encode the reference with this model's own VAE"
        )

    factors = getattr(latent_format, "latent_rgb_factors", None)
    reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)

    if requested == "raw":
        return LatentColorSpace.raw(channels, ndim,
                                    default_method="reinhard" if channels <= 4 else "mean_only")

    if factors is None:
        if requested == "rgb_proj":
            raise ValueError(
                f"{name} ships no latent_rgb_factors, so there is no RGB projection "
                f"to correct in; use space='raw'."
            )
        log.warning(
            "A1111 latent colour: %s has no latent_rgb_factors; falling back to raw "
            "per-channel matching (mean only, which cannot disturb structure).", name
        )
        return LatentColorSpace.raw(channels, ndim, default_method="mean_only",
                                    note="no rgb factors")

    inverse = None
    if reshape is not None:
        inverse = INVERSE_RESHAPES.get(name)
        if inverse is None:
            log.warning(
                "A1111 latent colour: %s re-packs its latent before projecting and we "
                "have no inverse for it; falling back to raw mean matching.", name
            )
            return LatentColorSpace.raw(channels, ndim, default_method="mean_only",
                                        note="un-invertible reshape")

    bias = getattr(latent_format, "latent_rgb_factors_bias", None)
    if measured_axes and reshape is None:
        measured = axes.lookup(name)
        if measured is not None and len(measured["factors"]) == channels:
            return LatentColorSpace.rgb(measured["factors"], measured["bias"], ndim=ndim,
                                        channels=channels, note="measured axes",
                                        tone=measured.get("tone"))
        log.info("A1111 latent colour: no measured decoder axes for %s; using the preview "
                 "matrix (colour will land off-hue and too strong on some families).", name)
    if channels == 3 and reshape is None and _is_identity(factors) and not bias:
        # Pixel-space formats (ChromaRadiance and friends) carry an identity
        # matrix: the latent already is RGB, so skip the projection entirely.
        return LatentColorSpace.pixel(ndim)

    return LatentColorSpace.rgb(factors, bias, reshape=reshape, inverse_reshape=inverse,
                                ndim=ndim, channels=channels)


def _is_identity(factors) -> bool:
    matrix = torch.as_tensor(factors, dtype=torch.float32)
    return matrix.shape == (3, 3) and torch.allclose(matrix, torch.eye(3), atol=1e-4)


def sigma_window(model, start_percent: float, end_percent: float):
    """Translate "from 20% to 80% of the schedule" into a sigma range.

    Working in sigma rather than step index keeps the setting meaningful
    whatever the user plugs in downstream — any sampler, scheduler, step
    count or denoise value — and across EPS, v-prediction and flow models,
    since every ModelSampling implements ``percent_to_sigma``.
    """
    sampling = model.get_model_object("model_sampling")
    return (float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)))


def fit_reference_dims(model, reference: torch.Tensor) -> torch.Tensor:
    """Give a still-image reference the time axis a video model expects.

    ``VAEEncode`` of a picture returns ``[B,C,H,W]`` whatever the model is, and
    a video latent format wants ``[B,C,T,H,W]``. Without this the scaling in
    ``process_in`` already fails for Wan, and for the simpler formats it would
    slip through and broadcast T against C. ComfyUI's own
    ``fix_empty_latent_channels`` does the same unsqueeze.
    """
    latent_format = model.get_model_object("latent_format")
    if getattr(latent_format, "latent_dimensions", 2) == 3 and reference.ndim == 4:
        return reference.unsqueeze(2)
    return reference


def detach(model, key: str = HOOK_KEY):
    """Return a clone with any guidance this pack applied removed."""
    patched = model.clone()
    try:
        import comfy.patcher_extension as patcher_extension
        patched.remove_wrappers_with_key(patcher_extension.WrappersMP.PREDICT_NOISE, key)
    except (ImportError, AttributeError):
        pass
    existing = patched.model_options.get("sampler_post_cfg_function", [])
    if existing:
        patched.model_options["sampler_post_cfg_function"] = [
            fn for fn in existing if getattr(fn, "_a1111_key", None) != key
        ]
    return patched


def attach(model, corrector, sigma_hi: float, sigma_lo: float, key: str = HOOK_KEY):
    """Clone the model and hang the correction off its noise prediction.

    ``WrappersMP.PREDICT_NOISE`` wraps ``CFGGuider.predict_noise``, which
    returns the denoised x0 estimate. Two reasons to prefer it over
    ``set_model_sampler_post_cfg_function``: wrappers are keyed, so applying
    the node twice replaces the hook instead of doubling its strength, and
    wrapping the whole call puts us after any post-CFG work other extensions
    do. On a ComfyUI too old to have it, the post-CFG hook is used instead —
    it reaches the same value by a narrower route.
    """
    patched = model.clone()
    run = {"last": None}

    def correct(x0, sigma):
        """Never let a colour correction take the whole render down with it."""
        # Sigmas fall within a run, so one that jumps back up starts a new run.
        starting = run["last"] is None or sigma > run["last"] * 1.05
        run["last"] = sigma
        if starting and sigma < sigma_lo:
            # The window is absolute on the model's schedule — ComfyUI's convention,
            # the same as ControlNet's start/end — so a low-denoise img2img can start
            # below all of it. With end_percent 0.9 a denoise of 0.3 on SDXL starts at
            # sigma 0.334 under an edge of 0.344 and not one step is corrected.
            # Measured the same day: end_percent 1.0 does reach such a run, but buys
            # little colour there (a*b* distance 0.20 -> 0.16) and halves high-frequency
            # energy at denoise 0.9, while A1111 Color Correction gets 0.12 for nothing.
            message = ("latent colour: this run starts at sigma %.3f, already below the "
                       "window's lower edge %.3f set by end_percent, so it will do nothing. "
                       "Low-denoise img2img starts that low; for such passes use A1111 "
                       "Color Correction on the decoded image (end_percent near 1.0 also "
                       "reaches them, at a cost in fine detail).")
            warn = getattr(corrector, "warn_once", None)
            if warn is not None:
                warn("below-window", message, sigma, sigma_lo)
            else:
                log.warning(message, sigma, sigma_lo)
        if not (sigma_lo <= sigma <= sigma_hi) or getattr(x0, "is_nested", False):
            return x0
        try:
            # Some correctors need to know where in the schedule they are —
            # a sigma that jumps back up means a new run has started.
            if getattr(corrector, "wants_sigma", False):
                return corrector(x0, sigma)
            return corrector(x0)
        except Exception as exc:  # noqa: BLE001 - a render in progress is worth more
            warn = getattr(corrector, "warn_once", None)
            if warn is None:
                log.warning("latent colour guidance failed: %r", exc)
            else:
                warn("failed",
                     "latent colour guidance turned itself off after an error: %r", exc)
            return x0

    try:
        import comfy.patcher_extension as patcher_extension
        wrapper_type = patcher_extension.WrappersMP.PREDICT_NOISE
    except (ImportError, AttributeError):
        log.warning(
            "A1111 latent colour: this ComfyUI has no predict_noise wrappers; "
            "using the post-CFG hook instead."
        )

        def post_cfg(args):
            return correct(args["denoised"], float(args["sigma"].flatten()[0]))

        post_cfg._a1111_key = key
        existing = patched.model_options.get("sampler_post_cfg_function", [])
        # That list is appended to, so without this a second application of the
        # node would simply apply the correction twice.
        patched.model_options["sampler_post_cfg_function"] = [
            fn for fn in existing if getattr(fn, "_a1111_key", None) != key
        ]
        patched.set_model_sampler_post_cfg_function(post_cfg)
        return patched

    def wrapper(executor, *args, **kwargs):
        x0 = executor(*args, **kwargs)
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        try:
            sigma = float(torch.as_tensor(timestep).flatten()[0])
        except Exception:  # noqa: BLE001 - never break sampling over a sigma read
            return x0
        return correct(x0, sigma)

    patched.remove_wrappers_with_key(wrapper_type, key)
    patched.add_wrapper_with_key(wrapper_type, key, wrapper)
    return patched
