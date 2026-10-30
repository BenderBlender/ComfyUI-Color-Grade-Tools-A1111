"""Structural checks on the example workflows.

A workflow that fails to open is worse than no example at all, so every file
in examples/ is checked for internal consistency, and every node type it uses
is checked to actually exist against a running ComfyUI when one is given.

    python tests/test_examples.py [--server 127.0.0.1:8188]

The widget rule is written here on its own, not imported from the generator:
the two once shared an inverted rule and agreed with each other while the
editor shifted IP-Adapter's values by one slot. A check that borrows the
checked code's assumptions cannot catch them.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")

# the only input types ComfyUI's editor turns into widgets
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def check_workflow(path, known_types=None):
    with open(path, encoding="utf-8") as fh:
        wf = json.load(fh)

    problems = []
    for key in ("nodes", "links", "last_node_id", "last_link_id", "version"):
        if key not in wf:
            problems.append(f"missing top-level key {key!r}")
    if problems:
        return problems

    by_id = {n["id"]: n for n in wf["nodes"]}
    if len(by_id) != len(wf["nodes"]):
        problems.append("duplicate node ids")

    seen_links = set()
    for link in wf["links"]:
        link_id, src_id, src_slot, dst_id, dst_slot, link_type = link
        if link_id in seen_links:
            problems.append(f"duplicate link id {link_id}")
        seen_links.add(link_id)

        src, dst = by_id.get(src_id), by_id.get(dst_id)
        if src is None or dst is None:
            problems.append(f"link {link_id} points at a missing node")
            continue
        if src_slot >= len(src["outputs"]):
            problems.append(f"link {link_id}: {src['type']} has no output {src_slot}")
            continue
        if dst_slot >= len(dst["inputs"]):
            problems.append(f"link {link_id}: {dst['type']} has no input {dst_slot}")
            continue

        out_type = src["outputs"][src_slot]["type"]
        in_type = dst["inputs"][dst_slot]["type"]
        if out_type != in_type:
            problems.append(
                f"link {link_id}: {src['type']}.{out_type} -> {dst['type']}.{in_type}")
        if dst["inputs"][dst_slot]["link"] != link_id:
            problems.append(f"link {link_id}: target input does not point back")
        if link_id not in src["outputs"][src_slot]["links"]:
            problems.append(f"link {link_id}: source output does not list it")

    for node in wf["nodes"]:
        if node["type"] == "MarkdownNote":
            continue
        if known_types is not None and node["type"] not in known_types:
            problems.append(f"unknown node type {node['type']!r}")
        for inp in node["inputs"]:
            link_id = inp.get("link")
            if link_id is not None and link_id not in seen_links:
                problems.append(f"{node['type']}.{inp['name']} points at missing link {link_id}")
        for inp in node["inputs"]:
            if "widget" in inp and inp.get("type") not in WIDGET_TYPES:
                problems.append(
                    f"{node['type']}.{inp['name']}: a {inp.get('type')} socket written as a "
                    "widget — the editor gives it no widget, so every value after it shifts")
        widgets = [i for i in node["inputs"] if "widget" in i]
        values = node.get("widgets_values", [])
        if len(values) < len(widgets):
            problems.append(
                f"{node['type']}: {len(widgets)} widgets but {len(values)} values")

    # every required socket input must be wired
    for node in wf["nodes"]:
        for inp in node["inputs"]:
            if "widget" in inp or inp.get("shape") == 7:
                continue
            if inp.get("link") is None:
                problems.append(f"{node['type']}.{inp['name']} is required but unwired")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="",
                    help="a running ComfyUI, to check that every node type exists there")
    args = ap.parse_args()

    known = None
    if args.server:
        import urllib.request
        with urllib.request.urlopen(f"http://{args.server}/object_info", timeout=60) as resp:
            known = json.load(resp)

    files = sorted(glob.glob(os.path.join(EXAMPLES, "*.json")))
    assert files, "no example workflows found"

    failed = 0
    for path in files:
        name = os.path.basename(path)
        problems = check_workflow(path, known)
        if problems:
            failed += 1
            print(f"- {name}: FAIL")
            for p in problems[:12]:
                print(f"    {p}")
        else:
            with open(path, encoding="utf-8") as fh:
                wf = json.load(fh)
            print(f"- {name}: ok ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")

    print(f"\n{len(files) - failed}/{len(files)} workflows valid")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
