"""Image-space colour maths shared by the nodes.

Everything here is plain numpy on ``uint8``/``float32`` arrays — no torch, no
ComfyUI imports — so it can be unit-tested on its own.

The reference being reproduced is ``modules/processing.py`` of
AUTOMATIC1111/stable-diffusion-webui::

    def apply_color_correction(correction, original_image):
        image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
            cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2LAB),
            correction, channel_axis=2), cv2.COLOR_LAB2RGB).astype("uint8"))
        image = blendLayers(image, original_image, BlendType.LUMINOSITY)
        return image.convert('RGB')

Note the argument order of ``blendmodes.blendLayers(background, foreground)``:
the histogram-matched image is the *backdrop* and the original generated image
is the *source*. Per the W3C compositing spec, ``luminosity`` keeps the
luminosity of the source and the hue/saturation of the backdrop — so A1111
takes **chroma from the corrected image and brightness from the original**.
Getting that round the wrong way inverts the whole operation, which is why
:func:`a1111_luminosity_blend` is written out explicitly below.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# blendmodes' _lum weights (non-linear sRGB; Rec.601), as the webui's pinned
# blendmodes==2022 has them. They were 0.3 / 0.59 / 0.11 here — the W3C
# compositing spec's — which put the blend up to 3/255 off the webui (30.10).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)
_EPS = 1e-6


def require_cv2():
    """Kept for callers that still want to know; nothing here depends on it."""
    return cv2 is not None


# --------------------------------------------------------------------------- #
# LAB
#
# OpenCV is used when it is installed, because A1111 runs on it and that makes
# the match exact. The numpy path below reproduces the same transform — the
# matrix, white point and 8-bit packing OpenCV documents — so the node pack
# needs nothing beyond numpy, which ComfyUI already requires. A node that
# cannot fail to install is worth the hundred lines.
# `tests/test_color.py` pins the two implementations to within a level or two
# of each other wherever OpenCV is importable.
# --------------------------------------------------------------------------- #

# OpenCV's sRGB -> XYZ matrix and D65 white point
_RGB2XYZ = np.array([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE = np.array([0.950456, 1.0, 1.088754], dtype=np.float64)

_DELTA = 0.008856          # (6/29)^3
_KAPPA = 903.3             # (29/3)^3


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _lab_f(t):
    return np.where(t > _DELTA, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _lab_f_inv(t):
    cube = t ** 3
    return np.where(cube > _DELTA, cube, (116.0 * t - 16.0) / _KAPPA)


def _rgb_to_lab_numpy(u8: np.ndarray) -> np.ndarray:
    rgb = _srgb_to_linear(u8.astype(np.float64) / 255.0)
    xyz = rgb @ _RGB2XYZ.T / _WHITE
    f = _lab_f(xyz)

    # 116*f(Y) - 16 reduces to KAPPA*Y on the lower branch, so one expression
    # covers both halves of the piecewise definition.
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    out = np.stack([lightness * 255.0 / 100.0, a + 128.0, b + 128.0], axis=-1)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def _lab_to_rgb_numpy(lab_u8: np.ndarray) -> np.ndarray:
    lab = lab_u8.astype(np.float64)
    lightness = lab[..., 0] * 100.0 / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xyz = np.stack([_lab_f_inv(fx), _lab_f_inv(fy), _lab_f_inv(fz)], axis=-1) * _WHITE
    rgb = _linear_to_srgb(xyz @ _XYZ2RGB.T)
    return np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8)


def rgb_to_lab(u8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> 8-bit LAB (L,a,b each packed into 0..255)."""
    if cv2 is not None:
        return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    return _rgb_to_lab_numpy(u8)


def lab_to_rgb(lab_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
    return _lab_to_rgb_numpy(lab_u8)


# --------------------------------------------------------------------------- #
# histogram matching
# --------------------------------------------------------------------------- #

def match_histograms_u8(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Channel-wise histogram matching for ``uint8 [H,W,C]`` arrays.

    Same construction as ``skimage.exposure.match_histograms(..., channel_axis=2)``
    — for every source level, the reference level with the same cumulative
    probability, found by linear interpolation — written out as a 256-entry LUT
    per channel so ``scikit-image`` is not needed to install this node pack.

    scikit-image writes its interpolated result into a uint8 array, which
    truncates; so does this. It used to round instead — «a deliberate
    difference» that left the node 1/255 off the webui on half the values and
    up to 16/255 after the blend (30.10, against the webui's own function on
    its pinned libraries). ``tests/test_color.py`` asserts equality with the
    real scikit-image when it is importable.
    """
    if src.ndim != 3 or ref.ndim != 3:
        raise ValueError("expected [H,W,C] arrays")
    if src.shape[-1] != ref.shape[-1]:
        raise ValueError("source and reference must have the same channel count")

    out = np.empty_like(src)
    n_src = src[..., 0].size
    n_ref = ref[..., 0].size

    for c in range(src.shape[-1]):
        src_counts = np.bincount(src[..., c].ravel(), minlength=256)
        ref_counts = np.bincount(ref[..., c].ravel(), minlength=256)
        src_levels = np.nonzero(src_counts)[0]
        ref_levels = np.nonzero(ref_counts)[0]

        src_cdf = np.cumsum(src_counts[src_levels]) / n_src
        ref_cdf = np.cumsum(ref_counts[ref_levels]) / n_ref

        lut = np.zeros(256, dtype=np.uint8)
        lut[src_levels] = np.interp(src_cdf, ref_cdf, ref_levels).clip(0, 255).astype(np.uint8)
        out[..., c] = lut[src[..., c]]
    return out


def match_histograms_masked_u8(src: np.ndarray, ref: np.ndarray,
                               src_mask: np.ndarray = None,
                               ref_mask: np.ndarray = None) -> np.ndarray:
    """:func:`match_histograms_u8`, but with the histograms gathered only from
    the pixels a mask selects. The LUT is still applied to the whole image, so
    the correction stays globally consistent while being *measured* on the
    region that matters — the usual want for inpainting, where the surrounding
    plate should not drag the statistics around.

    Masks are boolean ``[H,W]``. ``None`` means "use every pixel".
    """
    if src_mask is None and ref_mask is None:
        return match_histograms_u8(src, ref)

    out = np.empty_like(src)
    src_sel = src.reshape(-1, src.shape[-1]) if src_mask is None else src[src_mask]
    ref_sel = ref.reshape(-1, ref.shape[-1]) if ref_mask is None else ref[ref_mask]
    if src_sel.size == 0 or ref_sel.size == 0:
        return src.copy()

    for c in range(src.shape[-1]):
        src_counts = np.bincount(src_sel[..., c].ravel(), minlength=256)
        ref_counts = np.bincount(ref_sel[..., c].ravel(), minlength=256)
        src_levels = np.nonzero(src_counts)[0]
        ref_levels = np.nonzero(ref_counts)[0]
        if src_levels.size == 0 or ref_levels.size == 0:
            out[..., c] = src[..., c]
            continue

        src_cdf = np.cumsum(src_counts[src_levels]) / src_sel[..., c].size
        ref_cdf = np.cumsum(ref_counts[ref_levels]) / ref_sel[..., c].size

        # The mapping is only defined at levels the mask actually contains, so
        # the gaps are filled by interpolating along the level axis. Leaving
        # them as identity would put a step in the middle of the curve, which
        # shows up as banding wherever a soft mask edge spans it.
        mapped = np.interp(src_cdf, ref_cdf, ref_levels)
        lut = np.interp(np.arange(256), src_levels, mapped)      # truncated below, as unmasked
        out[..., c] = lut.clip(0, 255).astype(np.uint8)[src[..., c]]
    return out


# --------------------------------------------------------------------------- #
# blends
# --------------------------------------------------------------------------- #

def _luma(rgb_f: np.ndarray) -> np.ndarray:
    # in blendmodes' own order of operations: a matrix product rounds the last
    # bit differently, and after the truncation to uint8 that flipped 0.01 % of
    # values by one level
    return (rgb_f[..., 0] * _LUMA[0]) + (rgb_f[..., 1] * _LUMA[1]) + (rgb_f[..., 2] * _LUMA[2])


def _clip_color(c: np.ndarray) -> np.ndarray:
    """``blendmodes._clipColor``: pull out-of-gamut channels back towards the
    pixel's own luma instead of clamping them, which would shift its hue."""
    lum = _luma(c)[..., None]
    lo = c.min(axis=-1, keepdims=True)
    hi = c.max(axis=-1, keepdims=True)

    below = lo < 0.0
    if below.any():
        scaled = lum + (c - lum) * lum / np.maximum(lum - lo, _EPS)
        c = np.where(below, scaled, c)

    above = hi > 1.0
    if above.any():
        scaled = lum + (c - lum) * (1.0 - lum) / np.maximum(hi - lum, _EPS)
        c = np.where(above, scaled, c)
    return c


def _set_luma(rgb_f: np.ndarray, target_luma: np.ndarray) -> np.ndarray:
    """``blendmodes._setLum``: shift a colour to a given luma, keeping hue."""
    return _clip_color(rgb_f + (target_luma[..., None] - _luma(rgb_f)[..., None]))


def a1111_luminosity_blend(corrected_u8: np.ndarray, original_u8: np.ndarray) -> np.ndarray:
    """What A1111 actually does: chroma of ``corrected``, brightness of ``original``.

    A numpy port of ``blendmodes.blendLayers(corrected, original, LUMINOSITY)``
    as the webui pins it (blendmodes==2022): float64 in non-linear sRGB, then
    ``imageFloatToInt`` — clip to [0, 1], times 255, truncated to uint8.
    """
    bg = corrected_u8.astype(np.float64) / 255.0
    fg = original_u8.astype(np.float64) / 255.0
    out = _set_luma(bg, _luma(fg))
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def lab_luminosity_blend(top_u8: np.ndarray, base_u8: np.ndarray) -> np.ndarray:
    """L* of ``top``, a*/b* of ``base`` — the LAB analogue, kept as an option."""
    fused = rgb_to_lab(base_u8).copy()
    fused[..., 0] = rgb_to_lab(top_u8)[..., 0]
    return lab_to_rgb(fused)


def lab_color_blend(top_u8: np.ndarray, base_u8: np.ndarray) -> np.ndarray:
    """a*/b* of ``top``, L* of ``base`` — the LAB approximation of A1111's blend."""
    fused = rgb_to_lab(top_u8).copy()
    fused[..., 0] = rgb_to_lab(base_u8)[..., 0]
    return lab_to_rgb(fused)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #

def mix_u8(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """``(1-t)*a + t*b``. ``t`` is a scalar or an ``[H,W,1]`` weight map."""
    out = a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t
    return np.rint(out).clip(0, 255).astype(np.uint8)


def gaussian_blur_u8(img_u8: np.ndarray, radius: int) -> np.ndarray:
    """Blur with an odd kernel; even radii are rounded up rather than ignored.

    The first version tested ``radius % 2 == 1`` and quietly did nothing for
    every even value the slider could produce.
    """
    if radius <= 0:
        return img_u8
    k = radius | 1
    if cv2 is not None:
        return cv2.GaussianBlur(img_u8, (k, k), 0)

    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8          # OpenCV's own default
    half = k // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    out = img_u8.astype(np.float64)
    padded = np.pad(out, ((half, half), (0, 0), (0, 0)), mode="reflect")
    out = sum(kernel[i] * padded[i:i + out.shape[0]] for i in range(k))
    padded = np.pad(out, ((0, 0), (half, half), (0, 0)), mode="reflect")
    out = sum(kernel[i] * padded[:, i:i + out.shape[1]] for i in range(k))
    return np.rint(out).clip(0, 255).astype(np.uint8)


def resize_bilinear(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a float ``[H,W]`` map — used to fit a mask to its image."""
    if arr.shape == (height, width):
        return arr
    if cv2 is not None:
        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)

    src_h, src_w = arr.shape
    ys = (np.arange(height) + 0.5) * src_h / height - 0.5
    xs = (np.arange(width) + 0.5) * src_w / width - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, src_h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]

    top = arr[np.ix_(y0, x0)] * (1 - wx) + arr[np.ix_(y0, x1)] * wx
    bottom = arr[np.ix_(y1, x0)] * (1 - wx) + arr[np.ix_(y1, x1)] * wx
    return (top * (1 - wy) + bottom * wy).astype(arr.dtype)
