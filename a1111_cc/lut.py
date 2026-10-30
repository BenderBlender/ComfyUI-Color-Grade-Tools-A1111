"""Reading, applying and writing 3D colour lookup tables (.cube).

A LUT is a look someone else already made, or one of yours on its way out to
another programme. The format is plain text and the maths is trilinear
interpolation, so neither needs a dependency.

What can and cannot be written out is worth being clear about. A grade made
with the wheels is a fixed function of colour, so it bakes into a table
exactly. Matching to a reference is not: it depends on the picture being
matched, and a different picture would want a different table. The same goes
for the A1111 correction. Those stay nodes.
"""

import logging
import os
import re

import torch

log = logging.getLogger(__name__)

DEFAULT_SIZE = 33
MAX_SIZE = 64


class CubeError(ValueError):
    """A .cube file that cannot be read, with a reason a person can act on."""


def read_cube(path: str):
    """Parse a .cube file into ``(table, domain_min, domain_max)``.

    ``table`` is ``[N, N, N, 3]`` indexed as ``[b, g, r]`` — the order the
    format stores, with red moving fastest.
    """
    size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    values = []

    with open(path, encoding="utf-8", errors="replace") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("LUT_1D_SIZE"):
                raise CubeError(f"{os.path.basename(path)} is a 1D LUT; "
                                f"this reads 3D ones (LUT_3D_SIZE).")
            if upper.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if upper.startswith("DOMAIN_MIN"):
                domain_min = [float(v) for v in line.split()[1:4]]
                continue
            if upper.startswith("DOMAIN_MAX"):
                domain_max = [float(v) for v in line.split()[1:4]]
                continue

            parts = re.split(r"[\s,]+", line)
            if len(parts) < 3:
                raise CubeError(f"{os.path.basename(path)}, line {number}: "
                                f"expected three numbers, found {line!r}")
            try:
                values.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError as exc:
                raise CubeError(f"{os.path.basename(path)}, line {number}: "
                                f"{exc}") from exc

    if size is None:
        raise CubeError(f"{os.path.basename(path)} has no LUT_3D_SIZE line.")
    if size < 2 or size > 256:
        raise CubeError(f"{os.path.basename(path)}: a size of {size} is out of range.")
    if len(values) != size ** 3:
        raise CubeError(f"{os.path.basename(path)}: declared {size}^3 = {size ** 3} "
                        f"entries but found {len(values)}.")

    table = torch.tensor(values, dtype=torch.float32).reshape(size, size, size, 3)
    return table, torch.tensor(domain_min), torch.tensor(domain_max)


def apply_lut(rgb: torch.Tensor, table: torch.Tensor,
              domain_min=None, domain_max=None, strength: float = 1.0) -> torch.Tensor:
    """Apply a 3D LUT to ``[B, 3, H, W]`` by trilinear interpolation."""
    if strength <= 0.0:
        return rgb

    size = table.shape[0]
    table = table.to(device=rgb.device, dtype=rgb.dtype)

    source = rgb
    if domain_min is not None and domain_max is not None:
        lo = domain_min.to(rgb.device, rgb.dtype).reshape(1, 3, 1, 1)
        hi = domain_max.to(rgb.device, rgb.dtype).reshape(1, 3, 1, 1)
        source = (rgb - lo) / (hi - lo).clamp_min(1e-6)

    coords = source.clamp(0.0, 1.0) * (size - 1)
    base = coords.floor().long().clamp(0, size - 1)
    nxt = (base + 1).clamp(0, size - 1)
    frac = (coords - base).clamp(0.0, 1.0)

    r0, g0, b0 = base[:, 0], base[:, 1], base[:, 2]
    r1, g1, b1 = nxt[:, 0], nxt[:, 1], nxt[:, 2]
    fr = frac[:, 0].unsqueeze(-1)
    fg = frac[:, 1].unsqueeze(-1)
    fb = frac[:, 2].unsqueeze(-1)

    def corner(bi, gi, ri):
        return table[bi, gi, ri]

    c00 = corner(b0, g0, r0) * (1 - fr) + corner(b0, g0, r1) * fr
    c01 = corner(b0, g1, r0) * (1 - fr) + corner(b0, g1, r1) * fr
    c10 = corner(b1, g0, r0) * (1 - fr) + corner(b1, g0, r1) * fr
    c11 = corner(b1, g1, r0) * (1 - fr) + corner(b1, g1, r1) * fr

    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    out = (c0 * (1 - fb) + c1 * fb).movedim(-1, 1)

    return rgb + (out - rgb) * strength if strength < 1.0 else out


def identity_grid(size: int = DEFAULT_SIZE) -> torch.Tensor:
    """Every colour a table of this size addresses, as ``[1, 3, 1, N^3]``.

    Ordered the way .cube expects: red fastest, blue slowest.
    """
    axis = torch.linspace(0.0, 1.0, size)
    blue, green, red = torch.meshgrid(axis, axis, axis, indexing="ij")
    stacked = torch.stack([red.reshape(-1), green.reshape(-1), blue.reshape(-1)])
    return stacked.reshape(1, 3, 1, size ** 3)


def write_cube(path: str, table: torch.Tensor, size: int, title: str = "") -> None:
    """Write ``[1, 3, 1, N^3]`` of graded colours out as a .cube file."""
    flat = table.reshape(3, -1).transpose(0, 1).clamp(0.0, 1.0)
    if flat.shape[0] != size ** 3:
        raise ValueError(f"expected {size ** 3} entries, got {flat.shape[0]}")

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        if title:
            handle.write(f'TITLE "{title}"\n')
        handle.write(f"LUT_3D_SIZE {size}\n")
        handle.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        handle.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for row in flat.tolist():
            handle.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    os.replace(temporary, path)


def grade_to_cube(grade, size: int = DEFAULT_SIZE):
    """Bake a wheels grade into a table. Exact: it is a function of colour alone."""
    from . import grading as G

    size = max(2, min(int(size), MAX_SIZE))
    grid = identity_grid(size)
    return G.apply_grade(grid, grade).clamp(0.0, 1.0), size
