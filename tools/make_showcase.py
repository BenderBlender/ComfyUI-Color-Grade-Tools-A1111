"""Render the figures that show what each node does.

Split in two on purpose. The wheels, the presets, the scopes, the comparison
and reference matching are arithmetic on pixels: they run here, on the CPU,
and need nothing but this pack. The LUT figure and the sampling
demonstrations are done by the ComfyUI you already have open — start it the
usual way, with this pack in its custom_nodes, and the script talks to it:

    python tools/make_showcase.py --image PICTURE.png
    python tools/make_showcase.py --image PICTURE.png --reference LOOK.png \\
        --with-model CHECKPOINT.safetensors

The checkpoint is named the way the server's Load Checkpoint node lists it;
a bare file name or a full path is matched against that list. One
generation is queued at a time, waited for, and read back; the server's
queue, memory manager and Interrupt button all apply, and nothing in this
process touches the GPU. Every graph is checked against the server's
/object_info before the first one is queued.

    --dry-run     build and validate every graph, print them, queue nothing
    --gap 2.0     seconds to leave the GPU alone between generations
    --unload      ask the server to unload the model when done
    --server      where ComfyUI listens (default 127.0.0.1:8188)
    --size 768    side of the square the pictures are made at
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_api  # noqa: E402

CELL = 360

POSITIVE = "a harbour at dawn, boats, wet stone, photograph"
NEGATIVE = "blurry, low quality"


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def load_pack(root):
    """The pack, in this process, for the pixel arithmetic. The modules that
    need ComfyUI itself (the sampler-side nodes) do not load here and are
    not needed here; their import errors are expected and kept quiet."""
    logger = logging.getLogger("ColorKitPack")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        spec = importlib.util.spec_from_file_location(
            "ColorKitPack", os.path.join(root, "__init__.py"),
            submodule_search_locations=[root])
        module = importlib.util.module_from_spec(spec)
        sys.modules["ColorKitPack"] = module
        spec.loader.exec_module(module)
    finally:
        logger.setLevel(level)
    return module


def load_image_u8(path, size):
    """Centre-crop to a square and resize — the same pixels the server will
    read back from the upload, so the two sides agree exactly."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    short = min(img.size)
    left, top = (img.size[0] - short) // 2, (img.size[1] - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
    return np.asarray(img)


def to_tensor(u8):
    import torch
    return torch.from_numpy(np.ascontiguousarray(u8).astype(np.float32) / 255.0).unsqueeze(0)


def load_image(path, size):
    return to_tensor(load_image_u8(path, size))


def to_u8(image):
    return np.rint(image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


def sheet(panels, path, title="", cell=CELL, columns=None):
    """panels: (uint8 image, caption, sub-caption)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small = ImageFont.truetype("arial.ttf", 12)
        head = ImageFont.truetype("arialbd.ttf", 19)
    except OSError:
        font = small = head = ImageFont.load_default()

    columns = columns or len(panels)
    rows = (len(panels) + columns - 1) // columns
    pad, caption = 10, 42
    top = 36 if title else 0
    width = cell * columns + pad * (columns + 1)
    height = top + rows * (cell + caption) + pad

    canvas = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((pad + 2, 10), title, fill=(238, 238, 244), font=head)

    for index, (image, label, sub) in enumerate(panels):
        row, column = divmod(index, columns)
        x = pad + column * (cell + pad)
        y = top + row * (cell + caption) + pad
        picture = Image.fromarray(image)
        if picture.size != (cell, cell):
            picture = picture.resize((cell, int(cell * picture.height / picture.width)),
                                     Image.LANCZOS)
        canvas.paste(picture, (x, y))
        draw.text((x + 2, y + picture.height + 5), label, fill=(236, 236, 242), font=font)
        if sub:
            draw.text((x + 2, y + picture.height + 23), sub, fill=(146, 146, 156), font=small)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path, quality=90, optimize=True)
    print(f"  wrote {os.path.basename(path)}  {canvas.size}")


# --------------------------------------------------------------------------- #
# the figures that need no model, and no server
# --------------------------------------------------------------------------- #

def demo_wheels(pack, image, out):
    """Each wheel on its own, so the difference between them is visible."""
    print(" wheels")
    node = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
    base = {name: spec[1]["default"]
            for name, spec in node.INPUT_TYPES()["required"].items() if name != "image"}

    def graded(**changes):
        values = dict(base)
        values.update(changes)
        return to_u8(node.apply(image, **values)[0])

    # lift is an offset from zero; gain and gamma are absolute, around one.
    panels = [
        (to_u8(image), "original", "nothing applied"),
        (graded(lift_master=0.10), "lift 0.10", "shadows rise, white stays put"),
        (graded(gain_master=1.18), "gain 1.18", "highlights rise, black stays put"),
        (graded(gamma_master=1.25), "gamma 1.25", "midtones bend, both ends hold"),
    ]
    sheet(panels, os.path.join(out, "10-wheels-apart.png"),
          title="Each wheel does one thing, and leaves the other end of the range alone")

    panels = [
        (graded(lift_b=0.05, lift_r=-0.03), "lift towards blue", "cool shadows"),
        (graded(gain_r=0.06, gain_b=-0.04), "gain towards red", "warm highlights"),
        (graded(lift_b=0.05, lift_r=-0.03, gain_r=0.06, gain_b=-0.04),
         "both", "the split-tone a colourist reaches for first"),
    ]
    sheet(panels, os.path.join(out, "11-wheels-colour.png"),
          title="Colour on the shadows and the highlights separately")


def demo_presets(pack, image, out):
    print(" presets")
    from ColorKitPack.a1111_cc import presets as P
    node = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
    base = {name: spec[1]["default"]
            for name, spec in node.INPUT_TYPES()["required"].items() if name != "image"}

    panels = []
    for name, values in P.BUILT_IN.items():
        merged = dict(base)
        merged.update({k: v for k, v in values.items() if k in base})
        panels.append((to_u8(node.apply(image, **merged)[0]), name, ""))

    sheet(panels, os.path.join(out, "12-presets.png"), columns=5, cell=300,
          title="The looks that ship with it — starting points, not destinations")


def demo_scopes(pack, image, out):
    print(" scopes")
    grade = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
    scopes = pack.NODE_CLASS_MAPPINGS["ColorKitScopes"]()
    base = {name: spec[1]["default"]
            for name, spec in grade.INPUT_TYPES()["required"].items() if name != "image"}

    cast = dict(base)
    cast.update({"gain_r": 0.12, "gain_b": -0.09, "saturation": 1.25})
    pushed, = grade.apply(image, **cast)

    panels = []
    for picture, label in ((image, "original"), (pushed, "pushed warm, saturation 1.25")):
        drawn, report = scopes.measure(picture, mode="all", size=300)
        panels.append((to_u8(picture), label, report.split("\n")[1].strip()))
        panels.append((to_u8(drawn), f"scopes — {label}", ""))

    sheet(panels, os.path.join(out, "13-scopes.png"), columns=2, cell=560,
          title="What the scopes say that the picture does not")


def demo_compare(pack, image, out):
    print(" compare")
    grade = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
    compare = pack.NODE_CLASS_MAPPINGS["ColorKitCompare"]()
    base = {name: spec[1]["default"]
            for name, spec in grade.INPUT_TYPES()["required"].items() if name != "image"}

    subtle = dict(base)
    subtle.update({"gain_r": 0.022, "lift_b": 0.012})
    after, = grade.apply(image, **subtle)

    panels = [(to_u8(image), "before", ""), (to_u8(after), "after", "a very small grade")]
    for mode, note in (("wipe", "the line sits where you put it"),
                       ("difference", "amplified — the only honest look at a small change")):
        shown, = compare.compare(image, after, mode=mode, position=0.5, amplify=8.0)
        panels.append((to_u8(shown), mode, note))

    sheet(panels, os.path.join(out, "14-compare.png"), columns=4, cell=330,
          title="A change too small to see, made visible")


def demo_reference(pack, image, reference, out):
    print(" reference")
    node = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
    base = {name: spec[1]["default"]
            for name, spec in node.INPUT_TYPES()["required"].items() if name != "image"}

    matched, = node.apply(image, reference=reference, **base)
    warmed = dict(base)
    warmed.update({"gain_r": 0.05, "gain_b": -0.035})
    both, = node.apply(image, reference=reference, **warmed)

    sheet([(to_u8(image), "source", ""),
           (to_u8(reference), "reference", "the look to borrow"),
           (to_u8(matched), "matched", "wheels at rest"),
           (to_u8(both), "matched, then warmed", "the wheels work on top")],
          os.path.join(out, "15-reference.png"),
          title="Take the look from a picture, then adjust from there")


# --------------------------------------------------------------------------- #
# the figures the server makes
# --------------------------------------------------------------------------- #

def demo_luts(session, image_u8, image_name, out, size):
    """The LUT node reads its tables through ComfyUI's model folders, so it
    runs on the server — no model is involved, these are cheap prompts."""
    print(" luts")
    server = session.server
    if not server.reachable:
        print("   (no server, skipping)")
        return
    if server.object_info("ColorKitApplyLUT") is None:
        print("   (the server has no ColorKitApplyLUT node, skipping)")
        return
    available = server.combo_options("ColorKitApplyLUT", "lut_name")
    chosen = [n for n in available if n.startswith("ColorKit")][:3]
    if not chosen:
        print("   (no LUTs found on the server, skipping)")
        return

    panels = [(image_u8, "original", "")]
    for name in chosen:
        full = session.generate(comfy_api.graph_lut(image_name, name, 1.0),
                                f"16 luts / {name} at full", size, rest=False)
        half = session.generate(comfy_api.graph_lut(image_name, name, 0.5),
                                f"16 luts / {name} at half", size, rest=False)
        panels.append((full, os.path.splitext(name)[0], "at full strength"))
        panels.append((half, "", "at half"))

    sheet(panels, os.path.join(out, "16-luts.png"), columns=len(panels), cell=270,
          title="Any .cube loads, with a strength — film tables land heavily at full")


def demo_with_model(session, pack, image_u8, image_name, checkpoint, out, size=1024, steps=22):
    """The two places a grade can be applied, and what holding colour does."""
    gen = comfy_api.Generator(session, checkpoint, size, POSITIVE, NEGATIVE,
                              steps=steps, cfg=6.0, sampler="dpmpp_2m", scheduler="karras",
                              reference_image=image_name)
    image = to_tensor(image_u8)

    # 1. the same grade during sampling and after it
    image_node = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
    controls = {name: spec[1]["default"] for name, spec
                in image_node.INPUT_TYPES()["required"].items() if name != "image"}
    look = dict(controls)
    look.update({"lift_b": 0.045, "gain_r": 0.055, "gain_b": -0.040,
                 "saturation": 1.10})
    grade = {"strength": 0.6, "start_percent": 0.30, "end_percent": 0.95}
    grade.update(look)

    print(" grade during sampling vs after")
    plain = gen.txt2img("20 during-vs-after / no grade", seed=99001)
    during = gen.txt2img("20 during-vs-after / graded while sampling", seed=99001, grade=grade)

    after, = image_node.apply(to_tensor(plain), **look)

    sheet([(plain, "no grade", "the model left alone"),
           (to_u8(after), "graded afterwards", "a filter over a finished picture"),
           (during, "graded while sampling", "the model composed it this way")],
          os.path.join(out, "20-during-vs-after.png"),
          title="The same wheels, applied in two places")

    # 2. holding colour through an img2img pass
    print(" holding colour through img2img")
    drifted = gen.img2img("21 holding colour / img2img at 0.75", seed=4242, denoise=0.75)
    held = gen.img2img("21 holding colour / the same, colour held", seed=4242, denoise=0.75,
                       guidance=comfy_api.guidance(0.6, start=0.15, end=0.9))

    scopes = pack.NODE_CLASS_MAPPINGS["ColorKitScopes"]()
    sheet([(image_u8, "the source", "the colours to keep"),
           (drifted, "img2img at 0.75", "the palette has wandered"),
           (held, "the same, colour held", "same seed, same prompt")],
          os.path.join(out, "21-holding-colour.png"),
          title="An img2img pass drifts; the guidance node stops it")

    drift_scope, _ = scopes.measure(to_tensor(drifted), mode="vectorscope", size=360)
    held_scope, _ = scopes.measure(to_tensor(held), mode="vectorscope", size=360)
    source_scope, _ = scopes.measure(image, mode="vectorscope", size=360)
    sheet([(to_u8(source_scope), "source", ""),
           (to_u8(drift_scope), "drifted", "the cloud has moved"),
           (to_u8(held_scope), "held", "back where it started")],
          os.path.join(out, "22-holding-colour-scopes.png"),
          title="The same three, on a vectorscope")


def main():
    parser = argparse.ArgumentParser(
        description="Render the showcase figures; sampling through a running ComfyUI.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--out", default="docs/showcase")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--with-model", help="a checkpoint, as the server's Load Checkpoint lists it")
    parser.add_argument("--server", default=comfy_api.DEFAULT_ADDRESS,
                        help="host:port of the ComfyUI to use (default %(default)s)")
    parser.add_argument("--gap", type=float, default=2.0,
                        help="seconds to leave the GPU alone between generations "
                             "(default %(default)s)")
    parser.add_argument("--unload", action="store_true",
                        help="ask the server to unload the model when done")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate every graph against the server, print them, "
                             "queue nothing; the CPU figures go to a scratch folder")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    server = comfy_api.Server(args.server)
    if not server.ping():
        if args.with_model and not args.dry_run:
            raise SystemExit(server.unreachable_message())
        print(f"  {server.unreachable_message()}")
        print("  the pixel figures need no server; the LUT and sampling figures will be skipped")

    pack = load_pack(root)
    image_u8 = load_image_u8(args.image, args.size)
    image = to_tensor(image_u8)
    reference = load_image(args.reference, args.size) if args.reference else None
    image_name = f"colorkit_showcase_{args.size}.png"

    if args.dry_run:
        out = tempfile.mkdtemp(prefix="colorkit-showcase-dry-")
        print(f"  dry run: everything goes to {out}")

    started = time.time()
    demo_wheels(pack, image, out)
    demo_presets(pack, image, out)
    demo_scopes(pack, image, out)
    demo_compare(pack, image, out)
    if reference is not None:
        demo_reference(pack, image, reference, out)
    print(f"\n  no server needed so far: {time.time() - started:.0f}s")

    if not server.reachable and not args.dry_run:
        return

    checkpoint = None
    if server.reachable:
        try:
            resolved = comfy_api.preflight(server, {"model": args.with_model})
        except comfy_api.ServerError as exc:
            raise SystemExit(f"  {exc}") from None
        checkpoint = resolved.get("model")
    else:
        checkpoint = args.with_model

    session = comfy_api.Session(server, gap=args.gap)

    def server_figures(run_out):
        demo_luts(session, image_u8, image_name, run_out, args.size)
        if checkpoint:
            demo_with_model(session, pack, image_u8, image_name, checkpoint, run_out,
                            size=args.size)

    # Pass one: every graph built and checked, nothing queued. A typo fails here.
    scratch = tempfile.mkdtemp(prefix="colorkit-showcase-dry-")
    session.dry_run = True
    session.verbose = args.dry_run
    session.dump_dir = scratch if args.dry_run else None
    print("\nchecking every graph before anything is queued")
    server_figures(scratch)
    session.finish_dry()
    if args.dry_run:
        print(f"  dry run only; the graphs are in {scratch}")
        return
    shutil.rmtree(scratch, ignore_errors=True)

    # Pass two: for real.
    print(f"\nrendering to {out}")
    print(server.memory_line())
    server.upload_image(comfy_api.png_bytes(image_u8), image_name)
    session.dry_run = False
    session.built.clear()
    try:
        server_figures(out)
    except KeyboardInterrupt:
        print("\n  interrupted — the server was told to stop")
        raise SystemExit(130)
    finally:
        if args.unload:
            try:
                server.free(unload_models=True)
            except comfy_api.ServerError as exc:
                print(f"  could not ask the server to unload: {exc}")

    print(server.memory_line())
    print(f"\n  everything: {time.time() - started:.0f}s "
          f"({session.queued} generations, {session.seconds:.0f}s of them sampling)")


if __name__ == "__main__":
    main()
