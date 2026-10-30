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
        lift_b=0.085, lift_r=-0.050,
        gain_r=0.105, gain_b=-0.075,
        saturation=1.15, contrast=1.06,
    ),
    "Warm": _preset(gain_r=0.085, gain_b=-0.065, lift_r=0.030, saturation=1.06),
    "Cool": _preset(gain_b=0.085, gain_r=-0.065, lift_b=0.030, saturation=1.02),
    "Bleach bypass": _preset(saturation=0.38, contrast=1.38, gamma_master=0.90),
    "Film fade": _preset(lift_master=0.090, lift_b=0.045, gain_master=0.90,
                         saturation=0.78, contrast=0.88),
    "Golden hour": _preset(gain_r=0.100, gain_g=0.035, gain_b=-0.085,
                           gamma_master=1.08, saturation=1.12),
    "Moonlight": _preset(lift_b=0.080, gain_b=0.070, gain_r=-0.080,
                         gamma_master=0.90, saturation=0.65),

    # Written here rather than shipped as .cube files. A table bought or
    # downloaded is someone's work under someone's licence, and most "free"
    # packs forbid passing them on; naming one after a film borrows a
    # trademark as well. These are built from the wheels instead, which also
    # makes them better behaved: you can see where they are, take them apart,
    # and turn them down.
    "Green gold": _preset(
        # The green-and-amber treatment of a certain Montmartre cafe: warm
        # gold through the highlights, green sitting in the midtones, shadows
        # kept slightly cool so the green reads as colour and not as murk.
        lift_b=0.040, lift_g=0.020,
        gamma_g=0.060, gamma_r=0.025, gamma_b=-0.060,
        gain_r=0.090, gain_g=0.060, gain_b=-0.105,
        saturation=1.20, contrast=1.08,
    ),
    "Print stock": _preset(
        # The dense, contrasty look of a film print: deep shadows, rich reds,
        # highlights that roll rather than clip.
        lift_master=-0.035, lift_b=0.022,
        gamma_master=0.92,
        gain_r=0.065, gain_b=-0.028,
        saturation=1.28, contrast=1.26,
    ),
    "Two strip": _preset(
        # Early two-colour processes had no blue record at all: everything
        # lands between warm red and cyan.
        lift_b=0.045, lift_g=0.028,
        gain_r=0.120, gain_g=-0.025, gain_b=-0.070,
        saturation=0.60, contrast=1.10,
    ),
    "Day for night": _preset(
        # Shot in daylight, graded for dusk: cold, dim, and desaturated
        # without going monochrome. gain and gamma are absolute, around one --
        # writing -0.06 here once meant a gain of -0.06, and a black frame.
        lift_b=0.055, lift_r=-0.035,
        gamma_master=0.84, gamma_b=0.035,
        gain_master=0.82, gain_b=0.055, gain_r=-0.060,
        saturation=0.52, contrast=1.12,
    ),
    "Cross process": _preset(
        # Film run through the wrong chemistry: shadows go cyan, highlights
        # go yellow, and contrast climbs.
        lift_b=0.090, lift_g=0.035, lift_r=-0.050,
        gain_r=0.075, gain_g=0.065, gain_b=-0.110,
        saturation=1.35, contrast=1.30,
    ),
    "Crush blacks": _preset(lift_master=-0.055, contrast=1.18),
    "Lift shadows": _preset(lift_master=0.075, gamma_master=1.06),
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
    # cutting a character leaves the spaces either side of it; keep one
    name = " ".join(_SAFE_NAME.sub("", str(name)).split())
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
