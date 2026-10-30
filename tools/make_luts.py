"""Write the built-in looks out as .cube files.

These ship with the pack so that models/luts is not empty on a first run, and
so a look can be taken into Resolve, Premiere or another ComfyUI pack without
going through ComfyUI at all. They are generated from the same presets the
wheels use, so there is one source of truth rather than two.

Small on purpose: 17 points per axis, about 130 KB each. A table that size
samples a grade to within roughly 2% — fine for a look you are going to
adjust anyway. For an exact copy of a grade you have dialled in, use the
"Save Grade as LUT" node, which writes 33 points by default.

    python tools/make_luts.py [--size 17] [--out luts]
"""

import argparse
import importlib
import os
import sys
import types

SIZE = 17

# Neutral would be an identity table, which is of no use to anyone.
SKIP = {"Neutral"}


def load_pack():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    if "colorkit_pack" not in sys.modules:
        parent = types.ModuleType("colorkit_pack")
        parent.__path__ = [root]
        sys.modules["colorkit_pack"] = parent
    grading = importlib.import_module("colorkit_pack.a1111_nodes.grading")
    presets = importlib.import_module("a1111_cc.presets")
    lut = importlib.import_module("a1111_cc.lut")
    return grading, presets, lut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    grading, presets, lut = load_pack()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(root, "luts")
    os.makedirs(out, exist_ok=True)

    written = 0
    for name, values in presets.BUILT_IN.items():
        if name in SKIP:
            continue
        grade = grading._grade_from(values)
        table, used = lut.grade_to_cube(grade, args.size)

        filename = "ColorKit - " + name + ".cube"
        path = os.path.join(out, filename)
        lut.write_cube(path, table, used, title=f"ColorKit {name}")
        written += 1
        print(f"  {filename:34s} {used}^3  {os.path.getsize(path) / 1024:5.0f} KB")

    print(f"\n{written} written to {out}")


if __name__ == "__main__":
    main()
