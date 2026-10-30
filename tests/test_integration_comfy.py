"""Integration tests: the nodes driving a real checkpoint inside ComfyUI.

These need a working ComfyUI and at least one checkpoint, so they are kept
apart from the unit tests. Run with ComfyUI's own interpreter::

    python_embeded\\python.exe tests/test_integration_comfy.py \\
        --comfy  M:/ComfyUI-Easy-Install_9_work_1/ComfyUI-Easy-Install/ComfyUI \\
        --sd15   Q:/_models/checkpoints/SD/photon_v1.safetensors \\
        --image  path/to/reference.png \\
        --outdir results

What it checks:
  1. the guidance hook fires for every sampler, including the ones the old
     in-place callback silently did nothing for;
  2. an img2img pass that drifts in colour is pulled back towards its
     reference, without losing detail;
  3. the latent format is detected and the RGB projection is used;
  4. the same code path runs on a 16-channel model (SD3.5) as on a 4-channel one.
"""

import argparse
import os
import sys
import time

import numpy as np


def memory_report(tag=""):
    """Say what is resident. Loading checkpoints from a script bypasses
    ``main.py``, so nothing else reports it."""
    import torch
    parts = []
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        parts.append(f"VRAM {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB")
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM {(vm.total - vm.available) / 2**30:.0f}/{vm.total / 2**30:.0f} GB")
    except ImportError:
        pass
    # Returned rather than printed: the caller passes it to report(), which
    # both prints it and keeps it in the written-out log.
    return f"  [memory{' ' + tag if tag else ''}] " + "  ".join(parts)


def free_everything(tag=""):
    """Unload everything ComfyUI holds, including the CPU-side weights."""
    import gc

    import torch

    import comfy.model_management as mm

    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache(force=True)
    torch.cuda.empty_cache()
    return memory_report(f"after unload {tag}".strip())


def load_pack(comfy_root):
    """Import this node pack the way ComfyUI would."""
    import importlib.util

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "A1111CCPack", os.path.join(here, "__init__.py"), submodule_search_locations=[here]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["A1111CCPack"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #

def to_numpy(image):
    """ComfyUI IMAGE -> uint8 [H,W,3]."""
    return np.rint(image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


def colour_signature(img_u8):
    """Per-channel mean/std in LAB — what "the colours" means numerically."""
    import cv2
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)


def colour_distance(a_u8, b_u8):
    """Distance between two images' colour signatures, in LAB levels."""
    am, asd = colour_signature(a_u8)
    bm, bsd = colour_signature(b_u8)
    return float(np.abs(am - bm).mean()), float(np.abs(asd - bsd).mean())


def detail_energy(img_u8):
    """Mean gradient magnitude — a proxy for how much detail survived."""
    g = img_u8.astype(np.float32).mean(-1)
    gy, gx = np.gradient(g)
    return float(np.hypot(gx, gy).mean())


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def load_reference_image(path, size=512):
    import torch
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    short = min(img.size)
    left = (img.size[0] - short) // 2
    top = (img.size[1] - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def synthetic_reference(size=512):
    """A fallback plate with enough colour structure to measure against."""
    import torch
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    rgb = np.stack([0.25 + 0.6 * x, 0.35 + 0.4 * y, 0.7 - 0.45 * x * y], -1)
    rgb += 0.05 * np.sin(24 * np.pi * x)[..., None]
    return torch.from_numpy(np.clip(rgb, 0, 1).astype(np.float32)).unsqueeze(0)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_hook_fires_for_every_sampler(pack, model, positive, negative, latent, report):
    """The old node modified the sampler's tensor in place, which several
    samplers simply discard. The wrapper must fire for all of them."""
    import torch
    import nodes as comfy_nodes
    from A1111CCPack.a1111_cc import glue
    from A1111CCPack.a1111_cc.guidance import GuidanceConfig, LatentCorrector
    from A1111CCPack.a1111_cc.space import LatentColorSpace

    samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpmpp_2m",
                "dpmpp_sde", "ddim", "uni_pc", "lcm", "dpm_adaptive", "res_multistep"]

    rows = []
    for sampler in samplers:
        calls = {"n": 0}

        class Counting(LatentCorrector):
            def __call__(self, x0):
                calls["n"] += 1
                return x0

        space = LatentColorSpace.raw(latent["samples"].shape[1], 4)
        corrector = Counting(space, latent["samples"], GuidanceConfig("mean_only", 1.0))
        hi, lo = glue.sigma_window(model, 0.0, 1.0)
        patched = glue.attach(model, corrector, hi, lo)

        try:
            comfy_nodes.common_ksampler(patched, 1, 6, 7.0, sampler, "normal",
                                        positive, negative, latent, denoise=1.0)
            ok = calls["n"] > 0
        except Exception as exc:  # noqa: BLE001
            ok = False
            rows.append((sampler, 0, f"ERROR {type(exc).__name__}: {exc}"))
            continue
        rows.append((sampler, calls["n"], "ok" if ok else "HOOK NEVER FIRED"))

    report("\n  sampler coverage (hook calls per 6-step run):")
    failures = 0
    for name, n, note in rows:
        report(f"    {name:16s} {n:4d}  {note}")
        if note != "ok":
            failures += 1
    assert failures == 0, f"{failures} sampler(s) did not run the correction"
    return rows


def test_guidance_pulls_colour_back(pack, model, vae, positive, negative,
                                    reference_image, report, outdir, seed=12345):
    """An img2img pass drifts in colour; the guidance should pull it back."""
    import torch
    import nodes as comfy_nodes

    encode = comfy_nodes.VAEEncode()
    decode = comfy_nodes.VAEDecode()

    ref_latent, = encode.encode(vae, reference_image)
    ref_u8 = to_numpy(reference_image)

    def run(patched_model, tag):
        start = time.time()
        out, = comfy_nodes.common_ksampler(
            patched_model, seed, 24, 7.0, "dpmpp_2m", "karras",
            positive, negative, ref_latent, denoise=0.75,
        )
        img, = decode.decode(vae, out)
        u8 = to_numpy(img)
        if outdir:
            save_png(u8, os.path.join(outdir, f"{tag}.png"))
        return u8, time.time() - start

    plain_u8, t_plain = run(model, "02_img2img_no_guidance")

    guidance = pack.NODE_CLASS_MAPPINGS["ApplyLatentColorGuidance"]()
    guided_model, = guidance.apply(
        model, ref_latent, strength=0.6, start_percent=0.15, end_percent=0.9,
        method="auto", space="auto", max_gain=2.0, lowpass=0,
    )
    guided_u8, t_guided = run(guided_model, "03_img2img_with_guidance")

    if outdir:
        save_png(ref_u8, os.path.join(outdir, "01_reference.png"))

    plain_mean, plain_std = colour_distance(plain_u8, ref_u8)
    guided_mean, guided_std = colour_distance(guided_u8, ref_u8)

    report("\n  colour distance from the reference (LAB levels, lower is closer):")
    report(f"    without guidance   mean {plain_mean:6.2f}   spread {plain_std:6.2f}   ({t_plain:.1f}s)")
    report(f"    with guidance      mean {guided_mean:6.2f}   spread {guided_std:6.2f}   ({t_guided:.1f}s)")
    improvement = (plain_mean - guided_mean) / max(plain_mean, 1e-6) * 100
    report(f"    improvement        {improvement:5.1f}%")

    d_plain, d_guided = detail_energy(plain_u8), detail_energy(guided_u8)
    report(f"    detail energy      {d_plain:6.2f} -> {d_guided:6.2f} "
           f"({(d_guided - d_plain) / d_plain * 100:+.1f}%)")

    assert guided_mean < plain_mean, "guidance did not bring the colours closer"
    assert d_guided > d_plain * 0.8, "guidance cost more than 20% of the detail"
    return {
        "plain_mean": plain_mean, "guided_mean": guided_mean,
        "improvement_pct": improvement,
        "detail_plain": d_plain, "detail_guided": d_guided,
        "seconds_plain": t_plain, "seconds_guided": t_guided,
    }


def test_space_detection(pack, model, latent, report, expect_channels):
    from A1111CCPack.a1111_cc import glue

    space = glue.detect_space(model, latent["samples"], "auto")
    report(f"\n  detected space: {space}")
    assert space.channels == expect_channels, (space.channels, expect_channels)
    assert space.kind in ("rgb_proj", "pixel"), \
        f"expected the RGB projection, got {space.kind}"

    scaled = glue.to_model_space(model, latent["samples"])
    ratio = float(scaled.std() / latent["samples"].std())
    report(f"  process_in scale factor: {ratio:.5f}  "
           f"(the original node skipped this entirely)")
    return {"space": repr(space), "scale_ratio": ratio}


def test_corrector_on_this_format(pack, model, latent, report):
    """Run the correction against this model's real latent, without sampling.

    Cheap, and it covers the part that differs between model families: the
    scaling into model space, the projection, and the step back.
    """
    import torch
    from A1111CCPack.a1111_cc import glue
    from A1111CCPack.a1111_cc.guidance import GuidanceConfig, LatentCorrector

    samples = latent["samples"]
    space = glue.detect_space(model, samples, "auto")
    reference = glue.to_model_space(model, samples)

    # A latent whose colours are deliberately pushed away from the reference
    drifted = reference * 1.2 + 0.35
    corrector = LatentCorrector(space, reference, GuidanceConfig("reinhard", 1.0))
    corrected = corrector(drifted)

    before = float((space.forward(drifted).mean(dim=[2, 3])
                    - space.forward(reference).mean(dim=[2, 3])).abs().mean())
    after = float((space.forward(corrected).mean(dim=[2, 3])
                   - space.forward(reference).mean(dim=[2, 3])).abs().mean())
    report(f"  projected colour error {before:.4f} -> {after:.4f} "
           f"({(1 - after / max(before, 1e-9)) * 100:.1f}% removed)")

    assert torch.isfinite(corrected).all(), "correction produced non-finite values"
    assert after < before * 0.2, "the correction did not move the colours"
    return {"latent_error_before": before, "latent_error_after": after}


def test_loopback_chain(pack, model, vae, positive, negative, reference_image,
                        report, outdir, loops=3):
    """A real loopback chain: does the palette actually hold across passes?

    Deliberately small — 512px, a few passes — because the claim under test is
    about drift between passes, not about image quality.
    """
    import cv2
    import numpy as np

    node = pack.NODE_CLASS_MAPPINGS["A1111Loopback"]()
    ref_u8 = to_numpy(reference_image)

    def split(img_u8):
        a = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3).mean(0)
        b = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3).mean(0)
        d = np.abs(a - b)
        return float(d[0]), float((d[1] + d[2]) / 2)

    results = {}
    for label, correction, exposure in (("off", False, 0.0),
                                        ("a1111", True, 0.0),
                                        ("hold_exposure", True, 1.0)):
        passes, final = node.run(
            model=model, vae=vae, positive=positive, negative=negative,
            image=reference_image, seed=4242, steps=16, cfg=7.0,
            sampler_name="dpmpp_2m", scheduler="karras", loops=loops,
            denoise=0.65, final_denoise=0.45, denoise_curve="Linear",
            color_correction=correction, hold_exposure=exposure,
            keep_all_passes=True,
        )
        final_u8 = to_numpy(final)
        lightness, chroma = split(final_u8)
        results[label] = {"lightness": lightness, "chroma": chroma}
        if outdir:
            save_png(final_u8, os.path.join(outdir, f"loopback_{label}.png"))
            for i in range(passes.shape[0]):
                save_png(to_numpy(passes[i:i + 1]),
                         os.path.join(outdir, f"loopback_{label}_pass{i + 1}.png"))

    report(f"\n  loopback, {loops} passes, drift from the first image (LAB levels):")
    report(f"    {'mode':16s} {'lightness':>10s} {'chroma':>8s}")
    for label, values in results.items():
        report(f"    {label:16s} {values['lightness']:10.2f} {values['chroma']:8.2f}")

    assert results["a1111"]["chroma"] < results["off"]["chroma"], \
        "correction did not hold the colour"
    assert results["hold_exposure"]["lightness"] < results["a1111"]["lightness"], \
        "hold_exposure did not pin the lightness"
    return {"loopback": results}


def test_image_node_against_a_known_cast(pack, reference_image, report, outdir):
    """The image node should remove a deliberate cast and keep the brightness."""
    import torch

    node = pack.NODE_CLASS_MAPPINGS["A1111ColorCorrection"]()
    cast = reference_image.clone()
    cast[..., 0] = (cast[..., 0] * 1.25).clamp(0, 1)      # push red
    cast[..., 2] = (cast[..., 2] * 0.70).clamp(0, 1)      # pull blue

    corrected, = node.apply(cast, reference_image, strength=1.0, blend_mode="a1111",
                            match_channels="L+a+b", ref_blur=0)

    ref_u8, cast_u8, out_u8 = (to_numpy(t) for t in (reference_image, cast, corrected))
    if outdir:
        save_png(cast_u8, os.path.join(outdir, "10_cast.png"))
        save_png(out_u8, os.path.join(outdir, "11_cast_corrected.png"))

    before, _ = colour_distance(cast_u8, ref_u8)
    after, _ = colour_distance(out_u8, ref_u8)
    report("\n  image node, deliberate red/blue cast (LAB levels):")
    report(f"    before {before:6.2f}  ->  after {after:6.2f}   ({(1 - after / before) * 100:.1f}% removed)")

    luma = lambda u8: (u8.astype(np.float32) / 255) @ np.array([0.3, 0.59, 0.11], np.float32)
    luma_shift = float(np.abs(luma(out_u8) - luma(cast_u8)).mean())
    report(f"    brightness preserved to {luma_shift:.5f} (a1111 blend keeps the original's luma)")

    assert after < before * 0.35, "the cast was not substantially removed"
    assert luma_shift < 0.02, "the a1111 blend should not move brightness"
    return {"cast_before": before, "cast_after": after, "luma_shift": luma_shift}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def save_png(u8, path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(u8).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="path to the ComfyUI checkout")
    ap.add_argument("--sd15", help="a 4-channel checkpoint (SD1.5)")
    ap.add_argument("--sd3", help="a 16-channel checkpoint (SD3.x), optional")
    ap.add_argument("--image", help="reference image; a synthetic plate is used otherwise")
    ap.add_argument("--outdir", default="", help="where to write result images")
    ap.add_argument("--prompt", default="a photograph, natural light, fine detail")
    ap.add_argument("--loops", type=int, default=0,
                    help="also run a loopback chain of this many passes (0 = skip)")
    args = ap.parse_args()

    sys.path.insert(0, args.comfy)
    os.chdir(args.comfy)

    import torch
    import comfy.sd
    import nodes as comfy_nodes

    pack = load_pack(args.comfy)

    lines = []

    def report(text=""):
        print(text, flush=True)
        lines.append(text)

    reference = load_reference_image(args.image) if args.image else synthetic_reference()
    results = {}

    for label, ckpt, channels in (("SD1.5", args.sd15, 4), ("SD3.5", args.sd3, 16)):
        if not ckpt:
            continue
        report(f"\n{'=' * 68}\n{label}: {os.path.basename(ckpt)}\n{'=' * 68}")
        t0 = time.time()
        report(memory_report("before load"))
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
            ckpt, output_vae=True, output_clip=True
        )[:3]
        report(f"  loaded in {time.time() - t0:.1f}s")
        report(memory_report("after load"))

        latent, = comfy_nodes.VAEEncode().encode(vae, reference)
        section = {}
        section.update(test_space_detection(pack, model, latent, report, channels))
        section.update(test_corrector_on_this_format(pack, model, latent, report))

        if clip is None:
            # Several SD3.x releases ship without text encoders. The latent
            # path is what this run is here to prove, and that needs no prompt.
            report("  no text encoder in this checkpoint — skipping the sampling pass")
            results[label] = section
            model = clip = vae = None
            report(free_everything(label))
            continue

        text = comfy_nodes.CLIPTextEncode()
        positive, = text.encode(clip, args.prompt)
        negative, = text.encode(clip, "")

        if label == "SD1.5":
            test_hook_fires_for_every_sampler(pack, model, positive, negative, latent, report)
            section.update(test_image_node_against_a_known_cast(
                pack, reference, report, args.outdir))

        outdir = os.path.join(args.outdir, label) if args.outdir else ""
        section.update(test_guidance_pulls_colour_back(
            pack, model, vae, positive, negative, reference, report, outdir))
        if args.loops > 0:
            section.update(test_loopback_chain(
                pack, model, vae, positive, negative, reference, report, outdir,
                loops=args.loops))
        results[label] = section

        model = clip = vae = positive = negative = latent = None
        report(free_everything(label))

    report("\n" + "=" * 68)
    report("all integration checks passed")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        import json
        with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
