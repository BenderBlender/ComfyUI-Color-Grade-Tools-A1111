"""Tests for the package itself: it compiles, it imports, its keys are stable.

Runs under pytest, or directly::

    python tests/test_package.py

Why this file exists. ``__init__.py`` imports each node module inside a
try/except and only logs a failure, so that one broken node cannot hide the
rest. That is right in production and blind in testing: a syntax error in a
node module leaves the pack importable, just short of those nodes, and every
other test in this suite exercises ``a1111_cc`` rather than ``a1111_nodes`` —
so the whole suite passed for one commit while two node modules did not
compile at all. These tests import the modules DIRECTLY, with no safety net.

The node modules use relative imports (``from ..a1111_cc import ...``), so they
are imported through the top-level package the way ComfyUI loads it: by the
repository folder's own name, which contains hyphens and therefore can only be
reached through ``importlib``.

``loopback`` and ``legacy`` genuinely need ``comfy.samplers``; they are checked
for syntax but imported only when ComfyUI is on the path.
"""

import ast
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
PKG = os.path.basename(ROOT)

for _p in (PARENT, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FOLDERS = ["a1111_cc", "a1111_nodes"]

# modules that import comfy.* and so cannot be imported without ComfyUI
NEEDS_COMFY = {"a1111_nodes.loopback", "a1111_nodes.legacy"}

# The registry key is what a saved workflow stores. Renaming one silently
# breaks every graph that used it, so the list is pinned here: the display
# names carry the "ColorKit" prefix for searchability, the keys never change.
EXPECTED_KEYS = {
    "A1111ColorCorrection",
    "A1111LatentColorKSampler",
    "A1111Loopback",
    "ApplyLatentColorGuidance",
    "ColorKitApplyLUT",
    "ColorKitCompare",
    "ColorKitSaveGradeAsLUT",
    "ColorKitScopes",
    "ImageColorGrade",
    "LatentColorGrade",
    "LatentColorMatch",
}


def _sources():
    files = [os.path.join(ROOT, "__init__.py")]
    for folder in FOLDERS:
        base = os.path.join(ROOT, folder)
        for name in sorted(os.listdir(base)):
            if name.endswith(".py"):
                files.append(os.path.join(base, name))
    return files


def _submodules():
    out = []
    for folder in FOLDERS:
        base = os.path.join(ROOT, folder)
        for name in sorted(os.listdir(base)):
            if name.endswith(".py") and name != "__init__.py":
                out.append(f"{folder}.{name[:-3]}")
    return out


def test_every_source_compiles():
    """The failure this file was written for: a node module that will not parse."""
    broken = []
    for path in _sources():
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        try:
            compile(source, path, "exec")
        except SyntaxError as exc:
            broken.append(
                f"{os.path.relpath(path, ROOT)}:{exc.lineno}: {exc.msg}")
    assert not broken, "these files do not compile:\n  " + "\n  ".join(broken)


def test_every_module_imports():
    failed = []
    for name in _submodules():
        if name in NEEDS_COMFY:
            continue
        try:
            importlib.import_module(f"{PKG}.{name}")
        except Exception as exc:  # noqa: BLE001 - report all of them, not the first
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failed, "these modules do not import:\n  " + "\n  ".join(failed)


def test_comfy_modules_only_need_comfy():
    """loopback / legacy may fail ONLY on a missing comfy, never on their own code."""
    wrong = []
    for name in sorted(NEEDS_COMFY):
        try:
            importlib.import_module(f"{PKG}.{name}")
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] != "comfy":
                wrong.append(f"{name}: missing {exc.name}, which is not comfy")
        except Exception as exc:  # noqa: BLE001
            wrong.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not wrong, "\n  ".join(wrong)


def _node_classes():
    """Every class in the package that looks like a ComfyUI node, from the AST."""
    for path in _sources():
        rel = os.path.relpath(path, ROOT)
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            assigned = {}
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        name = getattr(target, "id", "")
                        if name:
                            assigned[name] = item.value
            methods = {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
            if "INPUT_TYPES" in methods or "INPUT_TYPES" in assigned:
                yield f"{rel}:{node.name}", assigned


def test_registry_keys_are_stable():
    """A renamed key breaks every saved workflow that used the node."""
    found = set()
    for path in _sources():
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if "NODE_CLASS_MAPPINGS" not in [
                    getattr(t, "id", "") for t in node.targets]:
                continue
            if isinstance(node.value, ast.Dict):
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        found.add(key.value)

    gone = EXPECTED_KEYS - found
    added = found - EXPECTED_KEYS
    assert not gone, (
        "these registry keys disappeared — every saved workflow using them "
        f"will break: {sorted(gone)}")
    assert not added, (
        f"new registry keys {sorted(added)} — add them to EXPECTED_KEYS once you "
        "are sure of the spelling, because it is what workflows will store")


def test_every_node_describes_itself():
    """Each node carries the fields ComfyUI needs, and a readable DESCRIPTION."""
    problems = []
    seen = 0
    for where, assigned in _node_classes():
        seen += 1
        for field in ("RETURN_TYPES", "FUNCTION", "CATEGORY"):
            if field not in assigned:
                problems.append(f"{where}: no {field}")

        if "DESCRIPTION" not in assigned:
            problems.append(f"{where}: no DESCRIPTION")
            continue
        try:
            text = ast.literal_eval(assigned["DESCRIPTION"])
        except ValueError:
            continue                          # built at runtime, not a literal
        if not isinstance(text, str) or len(text) < 40:
            problems.append(f"{where}: DESCRIPTION too short to help anyone")
            continue
        if text != text.strip():
            problems.append(f"{where}: DESCRIPTION starts or ends with blank space")
        if "\n\n\n" in text:
            problems.append(f"{where}: DESCRIPTION has a run of blank lines")

    assert seen >= 9, f"only found {seen} node classes — the AST walk missed some"
    assert not problems, "\n  ".join(problems)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"- {fn.__name__}")
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
