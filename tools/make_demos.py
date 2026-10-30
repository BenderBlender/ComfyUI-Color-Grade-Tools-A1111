"""Render the demonstration figures used in the README.

Everything the README claims is produced here, from real sampling runs, so
the numbers and the pictures come from the same place.

The sampling is done by the ComfyUI you already have open. Start it the
usual way, with this pack in its custom_nodes, then:

    python tools/make_demos.py --image reference.png --sdxl protovision.safetensors
    python tools/make_demos.py --image reference.png --sd3 sd35_medium_flan_fp8.safetensors

Checkpoints are named the way the server's Load Checkpoint node lists them;
a bare file name (or a full path) is matched against that list. The script
uploads the reference, queues one generation at a time, waits for it, and
reads the picture back. The server's queue, its memory manager and its
Interrupt button all apply, and nothing in this process touches the GPU:
only the arithmetic — distances, contact sheets, metrics.json — runs here,
on the CPU.

Before the first prompt is queued every graph is built and checked against
the server's /object_info, so a misspelt parameter fails in a second rather
than after twenty minutes of sampling.

    --dry-run     build and validate every graph, print them, queue nothing
    --gap 2.0     seconds to leave the GPU alone between generations
    --unload      ask the server to unload the models when done
    --server      where ComfyUI listens (default 127.0.0.1:8188)
    --out         where the figures go (default docs/demos)
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_api  # noqa: E402

PROMPT = ("two girls in summer dresses walking on a wooden pier by a lake at sunset, "
          "watercolour illustration, soft washes, warm golden light")
NEGATIVE = "photo, 3d render, harsh contrast, oversaturated"

# The sampler settings every figure in the README was made with.
SIZE = 1024
STEPS = 26
SEED = 7777
DENOISE = 0.72
CFG = 6.5
CFG_SD3 = 4.5
SAMPLER = "dpmpp_2m"
SCHEDULER = "karras"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def to_u8(image):
    return np.rint(image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


def to_tensor(u8):
    import torch
    return torch.from_numpy(np.ascontiguousarray(u8).astype(np.float32) / 255.0).unsqueeze(0)


def lab_signature(u8):
    import cv2
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    return lab.mean(0), lab.std(0)


def lab_distance(a_u8, b_u8):
    am, asd = lab_signature(a_u8)
    bm, bsd = lab_signature(b_u8)
    return float(np.abs(am - bm).mean()), float(np.abs(asd - bsd).mean())


def detail_energy(u8):
    g = u8.astype(np.float32).mean(-1)
    gy, gx = np.gradient(g)
    return float(np.hypot(gx, gy).mean())


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


def load_image(path, size):
    return to_tensor(load_image_u8(path, size))


def contact_sheet(panels, path, cell=420, title=""):
    """panels: list of (uint8 image, caption, sub-caption)."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 17)
        small = ImageFont.truetype("arial.ttf", 14)
        head = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = small = head = ImageFont.load_default()

    pad, caption_h = 10, 46
    top_h = 38 if title else 0
    width = cell * len(panels) + pad * (len(panels) + 1)
    height = top_h + cell + caption_h + pad * 2

    canvas = Image.new("RGB", (width, height), (22, 22, 26))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((pad + 2, 10), title, fill=(240, 240, 245), font=head)

    for i, (img_u8, caption, sub) in enumerate(panels):
        im = Image.fromarray(img_u8).resize((cell, cell), Image.LANCZOS)
        x = pad + i * (cell + pad)
        canvas.paste(im, (x, top_h + pad))
        y = top_h + pad + cell + 6
        draw.text((x + 2, y), caption, fill=(238, 238, 243), font=font)
        if sub:
            draw.text((x + 2, y + 21), sub, fill=(150, 150, 160), font=small)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)
    print(f"  wrote {os.path.basename(path)}  {canvas.size}")


def save(u8, path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(u8).save(path)


def load_pack(root):
    """The pack, in this process, for the image-side arithmetic. The modules
    that need ComfyUI itself (the sampler-side nodes) do not load here and
    are not needed here; their import errors are expected and kept quiet."""
    logger = logging.getLogger("A1111CCPack")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        spec = importlib.util.spec_from_file_location(
            "A1111CCPack", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
        pack = importlib.util.module_from_spec(spec)
        sys.modules["A1111CCPack"] = pack
        spec.loader.exec_module(pack)
    finally:
        logger.setLevel(level)
    return pack


# --------------------------------------------------------------------------- #
# the demonstrations
# --------------------------------------------------------------------------- #

def demo_blend_modes(gen, pack, reference, out, metrics):
    """Why the default blend had to change."""
    print(" demo 1: blend modes")
    node = pack.NODE_CLASS_MAPPINGS["A1111ColorCorrection"]()
    ref_u8 = to_u8(reference)

    drifted = gen.img2img("01 blend modes / img2img", seed=SEED, denoise=DENOISE)
    drifted_t = to_tensor(drifted)

    panels = [(ref_u8, "reference", "the colours to keep"),
              (drifted, "img2img result", f"drift {lab_distance(drifted, ref_u8)[0]:.1f}")]
    for mode, caption in (("a1111", "blend_mode = a1111"),
                          ("lab_luminosity", "the old default")):
        corrected, = node.apply(drifted_t, reference, strength=1.0, blend_mode=mode,
                                match_channels="L+a+b", ref_blur=0)
        u8 = to_u8(corrected)
        dist = lab_distance(u8, ref_u8)[0]
        note = "what A1111 does" if mode == "a1111" else "keeps the drift"
        panels.append((u8, caption, f"distance {dist:.1f} — {note}"))
        save(u8, os.path.join(out, "frames", f"blend_{mode}.png"))
        metrics[f"blend_{mode}"] = dist

    save(ref_u8, os.path.join(out, "frames", "reference.png"))
    save(drifted, os.path.join(out, "frames", "drifted.png"))
    metrics["blend_drifted"] = lab_distance(drifted, ref_u8)[0]
    contact_sheet(panels, os.path.join(out, "01-blend-modes.png"),
                  title="The blend the node applies decides whether the cast is removed at all")
    return drifted, drifted_t


def demo_strength(pack, reference, drifted_t, out, metrics):
    """The image node across its strength range."""
    print(" demo 2: image-node strength")
    node = pack.NODE_CLASS_MAPPINGS["A1111ColorCorrection"]()
    ref_u8 = to_u8(reference)
    panels = []
    for strength in (0.0, 0.35, 0.7, 1.0):
        corrected, = node.apply(drifted_t, reference, strength=strength,
                                blend_mode="a1111", match_channels="L+a+b", ref_blur=0)
        u8 = to_u8(corrected)
        dist = lab_distance(u8, ref_u8)[0]
        panels.append((u8, f"strength = {strength:g}", f"distance {dist:.1f}"))
        metrics[f"image_strength_{strength:g}"] = dist
    contact_sheet(panels, os.path.join(out, "02-image-strength.png"),
                  title="A1111 Color Correction — the whole range is useful, not just 1.0")


def demo_guidance(gen, reference, out, metrics):
    """Holding the colour during sampling rather than repairing it after."""
    print(" demo 3: latent guidance")
    ref_u8 = to_u8(reference)

    plain = gen.img2img("03 guidance / none", seed=SEED, denoise=DENOISE)
    panels = [(ref_u8, "reference", "the colours to keep"),
              (plain, "no guidance", f"distance {lab_distance(plain, ref_u8)[0]:.1f}")]
    metrics["guidance_off"] = lab_distance(plain, ref_u8)[0]
    metrics["guidance_off_detail"] = detail_energy(plain)
    save(plain, os.path.join(out, "frames", "guidance_off.png"))

    for strength in (0.4, 0.7):
        u8 = gen.img2img(f"03 guidance / {strength:g}", seed=SEED, denoise=DENOISE,
                         guidance=comfy_api.guidance(strength))
        dist = lab_distance(u8, ref_u8)[0]
        panels.append((u8, f"guidance {strength:g}", f"distance {dist:.1f}"))
        metrics[f"guidance_{strength:g}"] = dist
        metrics[f"guidance_{strength:g}_detail"] = detail_energy(u8)
        save(u8, os.path.join(out, "frames", f"guidance_{strength:g}.png"))

    contact_sheet(panels, os.path.join(out, "03-latent-guidance.png"),
                  title="Apply Latent Color Guidance — the palette is held while the image is built")


def demo_methods(gen, reference, out, metrics):
    """What the three transfer methods actually differ in."""
    print(" demo 4: methods")
    ref_u8 = to_u8(reference)
    panels = []
    for method in ("mean_only", "reinhard", "mkl"):
        u8 = gen.img2img(f"04 methods / {method}", seed=SEED, denoise=DENOISE,
                         guidance=comfy_api.guidance(0.6, method=method))
        dist = lab_distance(u8, ref_u8)
        panels.append((u8, f"method = {method}",
                       f"mean {dist[0]:.1f} · spread {dist[1]:.1f}"))
        metrics[f"method_{method}"] = dist[0]
        metrics[f"method_{method}_spread"] = dist[1]
    contact_sheet(panels, os.path.join(out, "04-methods.png"),
                  title="mean_only shifts, reinhard rescales, mkl also fixes casts across channels")


def demo_space(gen, reference, out, metrics):
    """Raw channels versus the RGB projection."""
    print(" demo 5: working space")
    ref_u8 = to_u8(reference)
    panels = []
    for space, caption in (("raw", "space = raw"), ("rgb_proj", "space = rgb_proj")):
        u8 = gen.img2img(f"05 working space / {space}", seed=SEED, denoise=DENOISE,
                         guidance=comfy_api.guidance(0.6, method="reinhard", space=space))
        dist = lab_distance(u8, ref_u8)[0]
        panels.append((u8, caption,
                       f"distance {dist:.1f} · detail {detail_energy(u8):.2f}"))
        metrics[f"space_{space}"] = dist
        metrics[f"space_{space}_detail"] = detail_energy(u8)
    contact_sheet(panels, os.path.join(out, "05-working-space.png"),
                  title="The projection moves only the colour directions; raw channels move everything")


def demo_sd35(gen, reference, out, section):
    """The same node on a 16-channel latent."""
    ref_u8 = to_u8(reference)
    plain = gen.img2img("06 sd35 / none", seed=SEED, denoise=DENOISE, cfg=CFG_SD3)
    guided = gen.img2img("06 sd35 / guidance 0.6", seed=SEED, denoise=DENOISE, cfg=CFG_SD3,
                         guidance=comfy_api.guidance(0.6))
    section["guidance_off"] = lab_distance(plain, ref_u8)[0]
    section["guidance_on"] = lab_distance(guided, ref_u8)[0]

    contact_sheet(
        [(ref_u8, "reference", "the colours to keep"),
         (plain, "SD3.5, no guidance", f"distance {section['guidance_off']:.1f}"),
         (guided, "SD3.5, guidance 0.6", f"distance {section['guidance_on']:.1f}")],
        os.path.join(out, "06-cross-model-sd35.png"),
        title="The same node on a 16-channel latent — no per-model configuration")
    save(plain, os.path.join(out, "frames", "sd35_off.png"))
    save(guided, os.path.join(out, "frames", "sd35_on.png"))


# --------------------------------------------------------------------------- #

def dump_metrics(out, metrics):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def figures(session, pack, reference, reference_name, checkpoints, out, metrics):
    """Every figure, in order. Runs once dry and once for real."""
    if checkpoints.get("SDXL"):
        print("SDXL (1024, 4-channel latent)")
        print(session.server.memory_line())
        gen = comfy_api.Generator(session, checkpoints["SDXL"], SIZE, PROMPT, NEGATIVE,
                                  steps=STEPS, cfg=CFG, sampler=SAMPLER, scheduler=SCHEDULER,
                                  reference_image=reference_name)
        section = {}
        try:
            drifted, drifted_t = demo_blend_modes(gen, pack, reference, out, section)
            demo_strength(pack, reference, drifted_t, out, section)
            demo_guidance(gen, reference, out, section)
            demo_methods(gen, reference, out, section)
            demo_space(gen, reference, out, section)
        finally:
            metrics["SDXL"] = section
            dump_metrics(out, metrics)          # survive an interruption

    if checkpoints.get("SD3.5"):
        print("SD3.5 (1024, 16-channel latent)")
        print(session.server.memory_line())
        gen = comfy_api.Generator(session, checkpoints["SD3.5"], SIZE, PROMPT, NEGATIVE,
                                  steps=STEPS, cfg=CFG_SD3, sampler=SAMPLER, scheduler=SCHEDULER,
                                  reference_image=reference_name)
        section = {}
        try:
            demo_sd35(gen, reference, out, section)
        finally:
            metrics["SD3.5"] = section
            dump_metrics(out, metrics)


def main():
    ap = argparse.ArgumentParser(
        description="Render the README's demonstration figures through a running ComfyUI.")
    ap.add_argument("--image", required=True, help="the reference picture")
    ap.add_argument("--sdxl", help="an SDXL checkpoint, as the server's Load Checkpoint lists it")
    ap.add_argument("--sd3", help="an SD3.5 checkpoint, likewise")
    ap.add_argument("--out", default="docs/demos")
    ap.add_argument("--server", default=comfy_api.DEFAULT_ADDRESS,
                    help="host:port of the ComfyUI to use (default %(default)s)")
    ap.add_argument("--gap", type=float, default=2.0,
                    help="seconds to leave the GPU alone between generations (default %(default)s)")
    ap.add_argument("--unload", action="store_true",
                    help="ask the server to unload its models when done")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate every graph against the server, print them, "
                         "queue nothing")
    args = ap.parse_args()
    if not (args.sdxl or args.sd3):
        ap.error("give --sdxl and/or --sd3")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    server = comfy_api.Server(args.server)
    if not server.ping():
        if not args.dry_run:
            raise SystemExit(server.unreachable_message())
        print(f"  {server.unreachable_message()}\n  dry run continues with structural checks only")

    pack = load_pack(here)
    reference_u8 = load_image_u8(args.image, SIZE)
    reference = to_tensor(reference_u8)
    reference_name = f"colorkit_reference_{SIZE}.png"

    wanted = {"SDXL": args.sdxl, "SD3.5": args.sd3}
    if server.reachable:
        try:
            checkpoints = comfy_api.preflight(server, wanted)
        except comfy_api.ServerError as exc:
            raise SystemExit(f"  {exc}") from None
    else:
        checkpoints = {k: v for k, v in wanted.items() if v}

    session = comfy_api.Session(server, gap=args.gap)

    # Pass one: every graph built and checked, nothing queued, pictures are
    # placeholders, figures go to a scratch folder. A typo fails here.
    scratch = tempfile.mkdtemp(prefix="colorkit-demos-dry-")
    session.dry_run = True
    session.verbose = args.dry_run
    session.dump_dir = scratch if args.dry_run else None
    print("checking every graph before anything is queued")
    figures(session, pack, reference, reference_name, checkpoints, scratch, {})
    session.finish_dry()
    if args.dry_run:
        print(f"  dry run only; the graphs are in {scratch}")
        return
    shutil.rmtree(scratch, ignore_errors=True)

    # Pass two: the same figures, for real.
    print(f"\nrendering to {out}")
    server.upload_image(comfy_api.png_bytes(reference_u8), reference_name)
    session.dry_run = False
    session.built.clear()
    metrics = {}
    started = time.time()
    try:
        figures(session, pack, reference, reference_name, checkpoints, out, metrics)
    except KeyboardInterrupt:
        print("\n  interrupted — the server was told to stop; metrics so far are in metrics.json")
        raise SystemExit(130)
    finally:
        dump_metrics(out, metrics)
        if args.unload:
            try:
                server.free(unload_models=True)
            except comfy_api.ServerError as exc:
                print(f"  could not ask the server to unload: {exc}")

    print(session.server.memory_line())
    print(f"  {session.queued} generations, {session.seconds:.0f}s sampling, "
          f"{time.time() - started:.0f}s in all")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
