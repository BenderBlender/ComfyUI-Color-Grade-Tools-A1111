"""Render the figures that show what each node does.

Split in two on purpose. Everything up to --with-model needs no checkpoint and
no GPU: the wheels, the presets, the scopes, the comparison and reference
matching are all arithmetic on pixels, and they are what most of the pack is.
The sampling demonstrations load one model, use it, and give the memory back.

    python tools/make_showcase.py --comfy PATH --image PICTURE.png
    python tools/make_showcase.py --comfy PATH --image PICTURE.png \\
        --with-model CHECKPOINT.safetensors --lean
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

CELL = 360


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def memory(tag=""):
    import torch
    bits = []
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        bits.append(f"VRAM {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB")
    try:
        import psutil
        vm = psutil.virtual_memory()
        bits.append(f"RAM {(vm.total - vm.available) / 2**30:.0f}/{vm.total / 2**30:.0f} GB")
    except ImportError:
        pass
    print(f"    [memory{' ' + tag if tag else ''}] " + "  ".join(bits), flush=True)


def free_everything(tag=""):
    import gc

    import torch

    import comfy.model_management as mm

    mm.unload_all_models()
    for name, kwargs in (("free_pins", {"evict_active": True, "loaded": True}),
                         ("cleanup_models_gc", {}), ("cleanup_models", {})):
        fn = getattr(mm, name, None)
        if fn is None:
            continue
        try:
            fn(1 << 62, **kwargs) if name == "free_pins" else fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"    [memory] {name}: {exc!r}")
    gc.collect()
    mm.soft_empty_cache(force=True)
    torch.cuda.empty_cache()
    memory(f"after unload {tag}".strip())


def load_pack(root):
    spec = importlib.util.spec_from_file_location(
        "ColorKitPack", os.path.join(root, "__init__.py"),
        submodule_search_locations=[root])
    module = importlib.util.module_from_spec(spec)
    sys.modules["ColorKitPack"] = module
    spec.loader.exec_module(module)
    return module


def load_image(path, size):
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    short = min(img.size)
    left, top = (img.size[0] - short) // 2, (img.size[1] - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).unsqueeze(0)


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
# the figures that need no model
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


def demo_luts(pack, image, out):
    print(" luts")
    node = pack.NODE_CLASS_MAPPINGS["ColorKitApplyLUT"]()
    available = node.INPUT_TYPES()["required"]["lut_name"][0]
    chosen = [n for n in available if n.startswith("ColorKit")][:3]
    if not chosen:
        print("   (no LUTs found, skipping)")
        return

    panels = [(to_u8(image), "original", "")]
    for name in chosen:
        full, = node.apply(image, name, 1.0)
        half, = node.apply(image, name, 0.5)
        panels.append((to_u8(full), os.path.splitext(name)[0], "at full strength"))
        panels.append((to_u8(half), "", "at half"))

    sheet(panels, os.path.join(out, "16-luts.png"), columns=len(panels), cell=270,
          title="Any .cube loads, with a strength — film tables land heavily at full")


# --------------------------------------------------------------------------- #

def demo_with_model(pack, image, checkpoint, out, size=1024, steps=22):
    """The two places a grade can be applied, and what holding colour does."""
    import comfy.sd
    import nodes as comfy_nodes
    import torch

    print(" loading the checkpoint")
    memory("before loading")
    started = time.time()
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        checkpoint, output_vae=True, output_clip=True)[:3]
    print(f"   loaded in {time.time() - started:.0f}s")
    memory("after loading")

    try:
        text = comfy_nodes.CLIPTextEncode()
        positive, = text.encode(clip, "a harbour at dawn, boats, wet stone, photograph")
        negative, = text.encode(clip, "blurry, low quality")

        latent, = comfy_nodes.VAEEncode().encode(vae, image)
        decode = comfy_nodes.VAEDecode()

        def sample(patched, source, denoise=1.0, seed=99001):
            out_latent, = comfy_nodes.common_ksampler(
                patched, seed, steps, 6.0, "dpmpp_2m", "karras",
                positive, negative, source, denoise=denoise)
            return to_u8(decode.decode(vae, out_latent)[0])

        empty, = comfy_nodes.EmptyLatentImage().generate(size, size, 1)

        # 1. the same grade during sampling and after it
        grade_node = pack.NODE_CLASS_MAPPINGS["LatentColorGrade"]()
        image_node = pack.NODE_CLASS_MAPPINGS["ImageColorGrade"]()
        controls = {name: spec[1]["default"] for name, spec
                    in image_node.INPUT_TYPES()["required"].items() if name != "image"}
        look = dict(controls)
        look.update({"lift_b": 0.045, "gain_r": 0.055, "gain_b": -0.040,
                     "saturation": 1.10})

        print(" grade during sampling vs after")
        plain = sample(model, empty)
        graded_model, = grade_node.apply(model, strength=0.6, start_percent=0.30,
                                         end_percent=0.95, **look)
        during = sample(graded_model, empty)

        plain_tensor = torch.from_numpy(plain.astype(np.float32) / 255).unsqueeze(0)
        after, = image_node.apply(plain_tensor, **look)

        sheet([(plain, "no grade", "the model left alone"),
               (to_u8(after), "graded afterwards", "a filter over a finished picture"),
               (during, "graded while sampling", "the model composed it this way")],
              os.path.join(out, "20-during-vs-after.png"),
              title="The same wheels, applied in two places")

        # 2. holding colour through an img2img pass
        print(" holding colour through img2img")
        drifted = sample(model, latent, denoise=0.75, seed=4242)
        guidance = pack.NODE_CLASS_MAPPINGS["ApplyLatentColorGuidance"]()
        held_model, = guidance.apply(model, latent, strength=0.6, start_percent=0.15,
                                     end_percent=0.9, method="auto", space="auto",
                                     max_gain=2.0, lowpass=0)
        held = sample(held_model, latent, denoise=0.75, seed=4242)

        scopes = pack.NODE_CLASS_MAPPINGS["ColorKitScopes"]()
        sheet([(to_u8(image), "the source", "the colours to keep"),
               (drifted, "img2img at 0.75", "the palette has wandered"),
               (held, "the same, colour held", "same seed, same prompt")],
              os.path.join(out, "21-holding-colour.png"),
              title="An img2img pass drifts; the guidance node stops it")

        drift_scope, _ = scopes.measure(
            torch.from_numpy(drifted.astype(np.float32) / 255).unsqueeze(0),
            mode="vectorscope", size=360)
        held_scope, _ = scopes.measure(
            torch.from_numpy(held.astype(np.float32) / 255).unsqueeze(0),
            mode="vectorscope", size=360)
        source_scope, _ = scopes.measure(image, mode="vectorscope", size=360)
        sheet([(to_u8(source_scope), "source", ""),
               (to_u8(drift_scope), "drifted", "the cloud has moved"),
               (to_u8(held_scope), "held", "back where it started")],
              os.path.join(out, "22-holding-colour-scopes.png"),
              title="The same three, on a vectorscope")

    finally:
        model = clip = vae = None
        free_everything("the checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--out", default="docs/showcase")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--with-model")
    parser.add_argument("--lean", action="store_true")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    sys.path.insert(0, args.comfy)
    os.chdir(args.comfy)

    if args.lean:
        import comfy.cli_args
        comfy.cli_args.args.disable_pinned_memory = True
        comfy.cli_args.args.disable_async_offload = True
        print("  lean mode: pinned memory and async offload off")

    pack = load_pack(root)
    image = load_image(args.image, args.size)
    reference = load_image(args.reference, args.size) if args.reference else None

    started = time.time()
    memory("before")
    demo_wheels(pack, image, out)
    demo_presets(pack, image, out)
    demo_scopes(pack, image, out)
    demo_compare(pack, image, out)
    demo_luts(pack, image, out)
    if reference is not None:
        demo_reference(pack, image, reference, out)
    print(f"\n  no model needed so far: {time.time() - started:.0f}s")
    memory("after the model-free figures")

    if args.with_model:
        demo_with_model(pack, image, args.with_model, out, size=args.size)
        print(f"\n  everything: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
