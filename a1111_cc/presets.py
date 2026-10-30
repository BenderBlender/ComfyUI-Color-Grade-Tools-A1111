"""Grade presets: a few to start from, and somewhere to keep your own.

Saved presets go in ComfyUI's user directory rather than inside this package,
so updating or reinstalling the nodes does not throw them away. The built-in
ones are read-only and always present.
"""

import json
import logging
import os
import re

log = logging.getLogger(__name__)

CONTROLS = (
    "lift_r", "lift_g", "lift_b", "lift_master",
    "gamma_r", "gamma_g", "gamma_b", "gamma_master",
    "gain_r", "gain_g", "gain_b", "gain_master",
    "saturation", "contrast",
)

NEUTRAL = {
    "lift_r": 0.0, "lift_g": 0.0, "lift_b": 0.0, "lift_master": 0.0,
    "gamma_r": 0.0, "gamma_g": 0.0, "gamma_b": 0.0, "gamma_master": 1.0,
    "gain_r": 0.0, "gain_g": 0.0, "gain_b": 0.0, "gain_master": 1.0,
    "saturation": 1.0, "contrast": 1.0,
}


def _preset(**changes):
    values = dict(NEUTRAL)
    values.update(changes)
    return values


# Deliberately gentle. A preset is a place to start from, not a look to
# accept: strong ones only teach you to turn them down.
BUILT_IN = {
    "Neutral": _preset(),
    "Teal and orange": _preset(
        lift_b=0.035, lift_r=-0.02,
        gain_r=0.045, gain_b=-0.03,
        saturation=1.08,
    ),
    "Warm": _preset(gain_r=0.035, gain_b=-0.025, lift_r=0.012),
    "Cool": _preset(gain_b=0.035, gain_r=-0.025, lift_b=0.012),
    "Bleach bypass": _preset(saturation=0.55, contrast=1.22, gamma_master=0.94),
    "Film fade": _preset(lift_master=0.045, lift_b=0.02, gain_master=0.96,
                         saturation=0.9, contrast=0.94),
    "Golden hour": _preset(gain_r=0.05, gain_g=0.015, gain_b=-0.04,
                           gamma_master=1.04, saturation=1.05),
    "Moonlight": _preset(lift_b=0.04, gain_b=0.03, gain_r=-0.035,
                         gamma_master=0.95, saturation=0.82),
}

_SAFE_NAME = re.compile(r"[^\w \-().]+", re.UNICODE)


def _user_file() -> str:
    """Where a user's own presets live."""
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 - usable outside ComfyUI, e.g. in tests
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".user")
    directory = os.path.join(base, "colorkit")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "grade_presets.json")


def clean_name(name: str) -> str:
    name = _SAFE_NAME.sub("", str(name)).strip()
    return name[:64]


def sanitise(values: dict) -> dict:
    """Keep only the controls we know, as floats, filling in the rest."""
    out = dict(NEUTRAL)
    for key in CONTROLS:
        if key in values:
            try:
                out[key] = float(values[key])
            except (TypeError, ValueError):
                pass
    return out


def load_user() -> dict:
    path = _user_file()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
        if not isinstance(stored, dict):
            raise ValueError("expected an object of name -> values")
        return {clean_name(k): sanitise(v) for k, v in stored.items()
                if clean_name(k) and isinstance(v, dict)}
    except Exception as exc:  # noqa: BLE001 - a broken file must not break the editor
        log.warning("ColorKit: could not read %s (%s); starting with none.", path, exc)
        return {}


def save_user(presets: dict) -> None:
    path = _user_file()
    tidy = {clean_name(k): sanitise(v) for k, v in presets.items() if clean_name(k)}
    # Written beside the target and moved into place, so an interrupted write
    # cannot leave a half-file where the presets used to be.
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(tidy, fh, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def all_presets() -> dict:
    """Built-ins first, then the user's — theirs win on a name clash."""
    merged = {f"· {name}": values for name, values in BUILT_IN.items()}
    merged.update(load_user())
    return merged
