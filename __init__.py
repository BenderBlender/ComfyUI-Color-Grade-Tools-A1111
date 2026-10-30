"""A1111 Color Correction — Automatic1111-style colour matching for ComfyUI.

Node modules each export their own ``NODE_CLASS_MAPPINGS`` /
``NODE_DISPLAY_NAME_MAPPINGS``; this file merges them so adding a node means
touching one list.
"""

import importlib
import logging

log = logging.getLogger(__name__)

__version__ = "1.0.0"

_MODULES = [
    "a1111_nodes.image_color_correction",
    "a1111_nodes.latent_nodes",
    "a1111_nodes.loopback",
    "a1111_nodes.grading",
    "a1111_nodes.inspect_nodes",
    "a1111_nodes.lut_nodes",
    "a1111_nodes.legacy",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _name in _MODULES:
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
    except Exception:  # noqa: BLE001 - one broken node must not hide the rest
        log.exception("A1111 Color Correction: failed to load %s", _name)
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

try:
    from .a1111_cc import routes as _routes
    _routes.register()
except Exception:  # noqa: BLE001 - the nodes must load even if the routes cannot
    log.exception("Color Grade: preset routes unavailable")

# Served to the editor: the colour-wheel widget for the grading nodes.
# The nodes work without it — it drives their ordinary FLOAT widgets.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
