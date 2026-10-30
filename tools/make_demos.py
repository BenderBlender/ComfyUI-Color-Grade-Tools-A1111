"""Render the demonstration figures used in the README.

Everything the README claims is produced here, from real sampling runs, so
the numbers and the pictures come from the same place.

    python_embeded\\python.exe tools/make_demos.py \\
        --comfy M:/ComfyUI-Easy-Install_9_work_1/ComfyUI-Easy-Install/ComfyUI \\
        --sdxl  "Q:/_models/checkpoints/SDXL/ProtoVision/protovision.safetensors" \\
        --sd3   "Q:/_models/checkpoints/SD35/sd35_medium_flan_fp8.safetensors" \\
        --image reference.png --out docs/demos
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

PROMPT = ("two girls in summer dresses walking on a wooden pier by a lake at sunset, "
          "watercolour illustration, soft washes, warm golden light")
NEGATIVE = "photo, 3d render, harsh contrast, oversaturated"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def memory_report(tag=""):
    """Print VRAM and system RAM, so a run says what it is costing.

    Loading checkpoints from a bare script bypasses ``main.py``, and nothing
    then reports what is resident. Measuring at every boundary is what tells
    you whether unloading actually worked rather than assuming it did.
    """
    import torch
    parts = []
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        parts.append(f"VRAM {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB")
        parts.append(f"torch reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB")
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM {(vm.total - vm.available) / 2**30:.0f}/{vm.total / 2**30:.0f} GB")
    except ImportError:
        pass
    print(f"    [memory{' ' + tag if tag else ''}] " + "  ".join(parts), flush=True)


def free_everything(tag=""):
    """Unload every model ComfyUI is holding and give the memory back."""
    import gc

    import torch

    import comfy.model_management as mm

    mm.unload_all_models()
    gc.collect()                      # drop the CPU-side weights too
    mm.soft_empty_cache(force=True)
    torch.cuda.empty_cache()
    memory_report(f"after unload {tag}".strip())


def to_u8(image):
    return np.rint(image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


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


def load_image(path, size):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    short = min(img.size)
    left, top = (img.size[0] - short) // 2, (img.size[1] - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
    import torch
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


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


# --------------------------------------------------------------------------- #

class Runner:
    def __init__(self, comfy_root, pack, ckpt, size, label):
        import comfy.sd
        import nodes as comfy_nodes

        self.pack = pack
        self.nodes = comfy_nodes
        self.label = label
        self.size = size

        memory_report(f"before loading {label}")
        t0 = time.time()
        self.model, self.clip, self.vae = comfy.sd.load_checkpoint_guess_config(
            ckpt, output_vae=True, output_clip=True)[:3]
        print(f"  {label}: loaded in {time.time() - t0:.0f}s")
        memory_report(f"after loading {label}")
        if self.clip is None:
            raise RuntimeError(f"{label}: this checkpoint has no text encoder")

        encode = comfy_nodes.CLIPTextEncode()
        self.positive, = encode.encode(self.clip, PROMPT)
        self.negative, = encode.encode(self.clip, NEGATIVE)

    def encode(self, image):
        latent, = self.nodes.VAEEncode().encode(self.vae, image)
        return latent

    def decode(self, latent):
        image, = self.nodes.VAEDecode().decode(self.vae, latent)
        return to_u8(image)

    def sample(self, model, latent, denoise=0.72, steps=26, seed=7777,
               sampler="dpmpp_2m", scheduler="karras", cfg=6.5):
        out, = self.nodes.common_ksampler(
            model, seed, steps, cfg, sampler, scheduler,
            self.positive, self.negative, latent, denoise=denoise)
        return out

    def guided(self, reference_latent, strength=0.6, method="auto",
               space="auto", start=0.15, end=0.9):
        node = self.pack.NODE_CLASS_MAPPINGS["ApplyLatentColorGuidance"]()
        model, = node.apply(self.model, reference_latent, strength=strength,
                            start_percent=start, end_percent=end, method=method,
                            space=space, max_gain=2.0, lowpass=0)
        return model

    def release(self):
        self.model = self.clip = self.vae = None
        self.positive = self.negative = None
        free_everything(self.label)


# --------------------------------------------------------------------------- #
# the demonstrations
# --------------------------------------------------------------------------- #

def demo_blend_modes(run, reference, out, metrics):
    """Why the default blend had to change."""
    print(" demo 1: blend modes")
    node = run.pack.NODE_CLASS_MAPPINGS["A1111ColorCorrection"]()
    ref_u8 = to_u8(reference)

    drifted = run.decode(run.sample(run.model, run.encode(reference)))
    import torch
    drifted_t = torch.from_numpy(drifted.astype(np.float32) / 255).unsqueeze(0)

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


def demo_strength(run, reference, drifted_t, out, metrics):
    """The image node across its strength range."""
    print(" demo 2: image-node strength")
    node = run.pack.NODE_CLASS_MAPPINGS["A1111ColorCorrection"]()
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


def demo_guidance(run, reference, out, metrics):
    """Holding the colour during sampling rather than repairing it after."""
    print(" demo 3: latent guidance")
    ref_u8 = to_u8(reference)
    ref_latent = run.encode(reference)

    plain = run.decode(run.sample(run.model, ref_latent))
    panels = [(ref_u8, "reference", "the colours to keep"),
              (plain, "no guidance", f"distance {lab_distance(plain, ref_u8)[0]:.1f}")]
    metrics["guidance_off"] = lab_distance(plain, ref_u8)[0]
    metrics["guidance_off_detail"] = detail_energy(plain)
    save(plain, os.path.join(out, "frames", "guidance_off.png"))

    for strength in (0.4, 0.7):
        u8 = run.decode(run.sample(run.guided(ref_latent, strength), ref_latent))
        dist = lab_distance(u8, ref_u8)[0]
        panels.append((u8, f"guidance {strength:g}", f"distance {dist:.1f}"))
        metrics[f"guidance_{strength:g}"] = dist
        metrics[f"guidance_{strength:g}_detail"] = detail_energy(u8)
        save(u8, os.path.join(out, "frames", f"guidance_{strength:g}.png"))

    contact_sheet(panels, os.path.join(out, "03-latent-guidance.png"),
                  title="Apply Latent Color Guidance — the palette is held while the image is built")


def demo_methods(run, reference, out, metrics):
    """What the three transfer methods actually differ in."""
    print(" demo 4: methods")
    ref_u8 = to_u8(reference)
    ref_latent = run.encode(reference)
    panels = []
    for method in ("mean_only", "reinhard", "mkl"):
        u8 = run.decode(run.sample(run.guided(ref_latent, 0.6, method=method), ref_latent))
        dist = lab_distance(u8, ref_u8)
        panels.append((u8, f"method = {method}",
                       f"mean {dist[0]:.1f} · spread {dist[1]:.1f}"))
        metrics[f"method_{method}"] = dist[0]
        metrics[f"method_{method}_spread"] = dist[1]
    contact_sheet(panels, os.path.join(out, "04-methods.png"),
                  title="mean_only shifts, reinhard rescales, mkl also fixes casts across channels")


def demo_space(run, reference, out, metrics):
    """Raw channels versus the RGB projection."""
    print(" demo 5: working space")
    ref_u8 = to_u8(reference)
    ref_latent = run.encode(reference)
    panels = []
    for space, caption in (("raw", "space = raw"), ("rgb_proj", "space = rgb_proj")):
        u8 = run.decode(run.sample(
            run.guided(ref_latent, 0.6, method="reinhard", space=space), ref_latent))
        dist = lab_distance(u8, ref_u8)[0]
        panels.append((u8, caption,
                       f"distance {dist:.1f} · detail {detail_energy(u8):.2f}"))
        metrics[f"space_{space}"] = dist
        metrics[f"space_{space}_detail"] = detail_energy(u8)
    contact_sheet(panels, os.path.join(out, "05-working-space.png"),
                  title="The projection moves only the colour directions; raw channels move everything")


# --------------------------------------------------------------------------- #

def dump_metrics(out, metrics):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--sdxl")
    ap.add_argument("--sd3")
    ap.add_argument("--sd15")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="docs/demos")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    sys.path.insert(0, args.comfy)
    os.chdir(args.comfy)

    spec = importlib.util.spec_from_file_location(
        "A1111CCPack", os.path.join(here, "__init__.py"), submodule_search_locations=[here])
    pack = importlib.util.module_from_spec(spec)
    sys.modules["A1111CCPack"] = pack
    spec.loader.exec_module(pack)

    metrics = {}

    if args.sdxl:
        print("SDXL (1024, 4-channel latent)")
        reference = load_image(args.image, 1024)
        run = Runner(args.comfy, pack, args.sdxl, 1024, "SDXL")
        section = {}
        try:
            from A1111CCPack.a1111_cc import glue
            section["space"] = repr(
                glue.detect_space(run.model, run.encode(reference)["samples"], "auto"))
            drifted, drifted_t = demo_blend_modes(run, reference, out, section)
            demo_strength(run, reference, drifted_t, out, section)
            demo_guidance(run, reference, out, section)
            demo_methods(run, reference, out, section)
            demo_space(run, reference, out, section)
        finally:
            metrics["SDXL"] = section
            dump_metrics(out, metrics)          # survive an interruption
            run.release()

    if args.sd3:
        print("SD3.5 (1024, 16-channel latent)")
        reference = load_image(args.image, 1024)
        run = Runner(args.comfy, pack, args.sd3, 1024, "SD3.5")
        section = {}
        try:
            from A1111CCPack.a1111_cc import glue
            ref_u8 = to_u8(reference)
            ref_latent = run.encode(reference)
            section["space"] = repr(
                glue.detect_space(run.model, ref_latent["samples"], "auto"))

            plain = run.decode(run.sample(run.model, ref_latent, cfg=4.5))
            guided = run.decode(
                run.sample(run.guided(ref_latent, 0.6), ref_latent, cfg=4.5))
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
        finally:
            metrics["SD3.5"] = section
            dump_metrics(out, metrics)
            run.release()

    dump_metrics(out, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
