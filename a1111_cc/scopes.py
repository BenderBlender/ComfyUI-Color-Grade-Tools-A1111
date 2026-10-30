"""Scopes: what the image is actually doing, drawn so you can read it.

Grading by eye on a monitor you have not profiled, against a preview that is
scaled and sitting next to a bright white node graph, is guesswork. These are
the three instruments that replace the guessing — the same three any grading
suite puts on screen.

* **Histogram** — how many pixels sit at each level, per channel. Tells you
  about exposure and about clipping.
* **Waveform** — brightness against horizontal position, so a column of the
  plot is a column of the picture. Tells you *where* the clipping is, and
  whether the sky is blowing out while the foreground sits in the mud.
* **Vectorscope** — hue and saturation, with no brightness at all. Tells you
  which way a cast runs, and the skin-tone line tells you whether faces have
  gone green.

Everything is plain torch, so there is no new dependency and it runs on
whatever device the image is already on.
"""

import math

import torch

# Rec.709, the same weights ComfyUI's own previews assume
LUMA = (0.2126, 0.7152, 0.0722)

BACKGROUND = 0.055
GRID = 0.16
LABEL = 0.55

# Where the six primary and secondary targets sit on a vectorscope, at 75%
# saturation — the boxes a colourist lines bars up against.
TARGETS = {
    "R": (1.0, 0.0, 0.0), "Yl": (1.0, 1.0, 0.0), "G": (0.0, 1.0, 0.0),
    "Cy": (0.0, 1.0, 1.0), "B": (0.0, 0.0, 1.0), "Mg": (1.0, 0.0, 1.0),
}


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """``[B,3,H,W]`` -> ``[B,H,W]``."""
    weights = torch.tensor(LUMA, device=rgb.device, dtype=rgb.dtype).reshape(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1)


def _chroma(rgb: torch.Tensor):
    """Rec.709 Cb and Cr, each in roughly [-0.5, 0.5]."""
    luma = _luma(rgb)
    return (rgb[:, 2] - luma) / 1.8556, (rgb[:, 0] - luma) / 1.5748


def _canvas(size, value=BACKGROUND):
    return torch.full((size, size, 3), value, dtype=torch.float32)


def _accumulate(flat_index: torch.Tensor, height: int, width: int) -> torch.Tensor:
    counts = torch.bincount(flat_index, minlength=height * width)
    return counts.reshape(height, width).float()


def _to_brightness(counts: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Counts to something the eye can read: log, because a scope spends most
    of its range on the few pixels that matter."""
    scaled = torch.log1p(counts * gain)
    peak = float(scaled.max())
    return scaled / peak if peak > 0 else scaled


def _draw_line(canvas, x0, y0, x1, y1, colour, width=1):
    """A straight line, drawn by walking it — no drawing library needed."""
    height, w = canvas.shape[:2]
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    tint = torch.tensor(colour, dtype=canvas.dtype)
    for i in range(steps):
        t = i / max(steps - 1, 1)
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        for dy in range(width):
            for dx in range(width):
                py, px = y + dy, x + dx
                if 0 <= py < height and 0 <= px < w:
                    canvas[py, px] = tint


def _draw_circle(canvas, cx, cy, radius, colour, steps=360):
    tint = torch.tensor(colour, dtype=canvas.dtype)
    height, width = canvas.shape[:2]
    for i in range(steps):
        angle = (i / steps) * math.tau
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        if 0 <= y < height and 0 <= x < width:
            canvas[y, x] = tint


def _draw_box(canvas, cx, cy, half, colour):
    _draw_line(canvas, cx - half, cy - half, cx + half, cy - half, colour)
    _draw_line(canvas, cx - half, cy + half, cx + half, cy + half, colour)
    _draw_line(canvas, cx - half, cy - half, cx - half, cy + half, colour)
    _draw_line(canvas, cx + half, cy - half, cx + half, cy + half, colour)


# --------------------------------------------------------------------------- #
# the three scopes
# --------------------------------------------------------------------------- #

def histogram(rgb: torch.Tensor, size: int = 512, bins: int = 256) -> torch.Tensor:
    """Pixel counts per level, one filled curve per channel."""
    canvas = _canvas(size)
    for fraction in (0.25, 0.5, 0.75):
        y = int(size * fraction)
        canvas[y, :] = GRID
        x = int(size * fraction)
        canvas[:, x] = GRID

    values = rgb.clamp(0, 1)
    counts = []
    for channel in range(3):
        quantised = (values[:, channel].reshape(-1) * (bins - 1)).long()
        counts.append(torch.bincount(quantised, minlength=bins).float())

    peak = max(float(c.max()) for c in counts) or 1.0
    column = torch.linspace(0, bins - 1, size).long()

    for channel, tint in enumerate(((0.95, 0.25, 0.25), (0.25, 0.95, 0.35),
                                    (0.35, 0.5, 1.0))):
        heights = (counts[channel][column] / peak * (size - 8)).long()
        colour = torch.tensor(tint, dtype=canvas.dtype)
        for x in range(size):
            top = size - 1 - int(heights[x])
            if top < size - 1:
                # Added rather than set, so overlapping channels read as grey
                # where the image is neutral — which is the useful signal.
                canvas[top:size - 1, x] = (canvas[top:size - 1, x] + colour * 0.5).clamp(0, 1)
    return canvas


def waveform(rgb: torch.Tensor, size: int = 512, parade: bool = False) -> torch.Tensor:
    """Brightness against horizontal position: a column here is a column there."""
    canvas = _canvas(size)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = min(size - 1, int(size * (1.0 - fraction)))
        canvas[y, :] = GRID

    image = rgb.clamp(0, 1)
    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(
            image, size=(min(image.shape[-2], 512), size), mode="area")

    panels = ([(image[:, c], tint) for c, tint in
               enumerate(((0.95, 0.3, 0.3), (0.3, 0.95, 0.4), (0.4, 0.55, 1.0)))]
              if parade else [(_luma(image), (0.85, 0.9, 0.95))])

    width = size // len(panels)
    for index, (plane, tint) in enumerate(panels):
        flat = plane.reshape(-1, plane.shape[-1])
        levels = (flat.clamp(0, 1) * (size - 1)).long()
        columns = torch.arange(flat.shape[-1], device=flat.device)
        columns = columns.expand(flat.shape[0], -1)

        target_column = (columns.reshape(-1).float() * (width - 1)
                         / max(flat.shape[-1] - 1, 1)).long() + index * width
        rows = (size - 1) - levels.reshape(-1)
        counts = _accumulate(rows * size + target_column.clamp(0, size - 1), size, size)

        brightness = _to_brightness(counts, gain=4.0)
        colour = torch.tensor(tint, dtype=canvas.dtype).reshape(1, 1, 3)
        canvas += brightness.unsqueeze(-1) * colour
    return canvas.clamp(0, 1)


def vectorscope(rgb: torch.Tensor, size: int = 512) -> torch.Tensor:
    """Hue and saturation with brightness discarded, plus the usual graticule."""
    canvas = _canvas(size)
    centre = size / 2
    radius = size * 0.44

    blue_diff, red_diff = _chroma(rgb.clamp(0, 1))
    x = (centre + blue_diff.reshape(-1) * radius * 2).clamp(0, size - 1).long()
    y = (centre - red_diff.reshape(-1) * radius * 2).clamp(0, size - 1).long()
    counts = _accumulate(y * size + x, size, size)

    brightness = _to_brightness(counts, gain=8.0)
    canvas += brightness.unsqueeze(-1) * torch.tensor([0.85, 0.95, 0.85]).reshape(1, 1, 3)

    for fraction in (0.25, 0.5, 0.75, 1.0):
        _draw_circle(canvas, centre, centre, radius * fraction, (GRID, GRID, GRID))
    _draw_line(canvas, centre - radius, centre, centre + radius, centre, (GRID,) * 3)
    _draw_line(canvas, centre, centre - radius, centre, centre + radius, (GRID,) * 3)

    for name, colour in TARGETS.items():
        patch = torch.tensor(colour, dtype=torch.float32).reshape(1, 3, 1, 1) * 0.75
        cb, cr = _chroma(patch)
        tx = centre + float(cb) * radius * 2
        ty = centre - float(cr) * radius * 2
        _draw_box(canvas, tx, ty, 7, (LABEL, LABEL, LABEL))

    # The skin-tone line: faces of every complexion land along it, which makes
    # it the fastest check there is for a cast that has gone green or magenta.
    angle = math.radians(123.0)
    _draw_line(canvas, centre, centre,
               centre + math.cos(angle) * radius, centre - math.sin(angle) * radius,
               (0.75, 0.55, 0.45))
    return canvas.clamp(0, 1)


def clipping(rgb: torch.Tensor, threshold: float = 0.004):
    """How much of the picture has hit the ends of the range, per channel."""
    low = (rgb <= threshold).float().mean(dim=(0, 2, 3))
    high = (rgb >= 1.0 - threshold).float().mean(dim=(0, 2, 3))
    return [float(v) for v in low], [float(v) for v in high]


def compose(rgb: torch.Tensor, mode: str = "all", size: int = 512) -> torch.Tensor:
    """One scope, or all three side by side."""
    if mode == "histogram":
        return histogram(rgb, size)
    if mode == "waveform":
        return waveform(rgb, size)
    if mode == "parade":
        return waveform(rgb, size, parade=True)
    if mode == "vectorscope":
        return vectorscope(rgb, size)

    panels = [vectorscope(rgb, size), waveform(rgb, size, parade=True),
              histogram(rgb, size)]
    gap = 6
    width = size * len(panels) + gap * (len(panels) - 1)
    canvas = torch.full((size, width, 3), BACKGROUND * 0.6, dtype=torch.float32)
    for index, panel in enumerate(panels):
        left = index * (size + gap)
        canvas[:, left:left + size] = panel
    return canvas
