"""Loading a .cube LUT, and baking a grade into one.

Two directions. Inward: a look someone else made — film emulation, a print
stock, a pack you bought — applied to an image. Outward: the grade you just
dialled in on the wheels, written as a table that Resolve, Premiere, Affinity
or another ComfyUI pack can read.

Only the wheels bake. Matching to a reference and the A1111 correction depend
on the picture being corrected, so there is no fixed table that represents
them; those stay as nodes.
"""

import logging
import os

import torch

from ..a1111_cc import lut as L
from .grading import GRADE_CONTROLS, _grade_from

log = logging.getLogger(__name__)

FOLDER = "luts"
EXTENSIONS = {".cube"}


def _register_folder():
    """Teach ComfyUI about models/luts, and look inside this pack as well."""
    try:
        import folder_paths
    except ImportError:
        return None

    base = os.path.join(folder_paths.models_dir, FOLDER)
    os.makedirs(base, exist_ok=True)
    try:
        folder_paths.add_model_folder_path(FOLDER, base)
    except Exception:  # noqa: BLE001 - older ComfyUI, or already registered
        folder_paths.folder_names_and_paths.setdefault(FOLDER, ([], EXTENSIONS))
        if base not in folder_paths.folder_names_and_paths[FOLDER][0]:
            folder_paths.folder_names_and_paths[FOLDER][0].append(base)

    shipped = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts")
    if os.path.isdir(shipped):
        paths = folder_paths.folder_names_and_paths[FOLDER][0]
        if shipped not in paths:
            paths.append(shipped)
    return base


LUT_DIR = _register_folder()


def _available():
    try:
        import folder_paths
        names = folder_paths.get_filename_list(FOLDER)
    except Exception:  # noqa: BLE001
        names = []
    return list(names) or ["(put .cube files in models/luts)"]


def _resolve(name):
    import folder_paths
    return folder_paths.get_full_path(FOLDER, name)


class ColorKitApplyLUT:
    """Apply a 3D lookup table to an image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_name": (_available(), {
                    "tooltip": "A .cube file from models/luts. Film emulation packs, "
                               "print-stock tables and anything else in that format.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mix between the original and the table's result. "
                               "Most film LUTs are built for log footage and land "
                               "heavily on an ordinary image — half strength is "
                               "often closer to what you want.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "ColorKit/image"
    DESCRIPTION = "Apply a .cube lookup table, with a strength to dial it back."

    def apply(self, image, lut_name, strength=1.0):
        if strength <= 0.0:
            return (image,)

        path = _resolve(lut_name)
        if not path or not os.path.exists(path):
            log.warning("ColorKit LUT: %r not found; passing the image through.", lut_name)
            return (image,)

        try:
            table, low, high = L.read_cube(path)
        except L.CubeError as exc:
            log.warning("ColorKit LUT: %s", exc)
            return (image,)

        planes = image.detach().float().movedim(-1, 1)[:, :3].clamp(0.0, 1.0)
        out = L.apply_lut(planes, table, low, high, float(strength)).movedim(1, -1)

        if image.shape[-1] == 4:
            out = torch.cat([out, image[..., 3:]], dim=-1)
        return (out.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0),)


class ColorKitSaveGradeAsLUT:
    """Write the wheels out as a .cube, to use anywhere else."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "filename": ("STRING", {
                "default": "my_grade",
                "tooltip": "Written into models/luts, with .cube added.",
            }),
            "size": ("INT", {
                "default": 33, "min": 2, "max": 64, "step": 1,
                "tooltip": "Points per axis. 33 is the usual choice and is plenty "
                           "for a grade; 17 is lighter, 64 is overkill.",
            }),
        }
        required.update(GRADE_CONTROLS)
        return {"required": required, "optional": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ColorKit/image"
    DESCRIPTION = (
        "Bake the wheels into a .cube file. Exact, because a grade is a function "
        "of colour alone — unlike matching to a reference, which depends on the "
        "picture and so has no fixed table."
    )

    def save(self, filename, size, image=None, **controls):
        grade = _grade_from(controls)
        table, used = L.grade_to_cube(grade, int(size))

        safe = "".join(c for c in str(filename) if c.isalnum() or c in " _-.()").strip()
        safe = safe or "my_grade"
        if not safe.lower().endswith(".cube"):
            safe += ".cube"

        directory = LUT_DIR or os.getcwd()
        path = os.path.join(directory, safe)
        try:
            L.write_cube(path, table, used, title=os.path.splitext(safe)[0])
            log.info("ColorKit: wrote %s (%d^3)", path, used)
        except OSError as exc:
            log.exception("ColorKit: could not write the LUT")
            return (image if image is not None else torch.zeros(1, 1, 1, 3), str(exc))

        if image is None:
            image = torch.zeros(1, 1, 1, 3)
        return (image, path)


NODE_CLASS_MAPPINGS = {
    "ColorKitApplyLUT": ColorKitApplyLUT,
    "ColorKitSaveGradeAsLUT": ColorKitSaveGradeAsLUT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorKitApplyLUT": "ColorKit Apply LUT (.cube)",
    "ColorKitSaveGradeAsLUT": "ColorKit Save Grade as LUT (.cube)",
}
