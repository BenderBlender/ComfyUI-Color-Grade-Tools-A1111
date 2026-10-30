"""Measure a model family's decoder colour axes for a1111_cc/axes.py, through the ComfyUI already running.

ComfyUI's ``latent_rgb_factors`` are a fit of decoded pictures to latents, made for previews. Inverting
that fit to WRITE colour into a latent is what the grade wheels do, and on SDXL the fit's colour columns
are so nearly dependent (condition number 11) that "+red" came back as red minus green, darker and three
to five times too strong. What the wheels need is the decoder's response to a change, measured here:

  1. the picture is VAE-encoded on the server and the latent fetched;
  2. J = d(decoded RGB) / d(model-space latent): the latent is nudged by +-delta along each channel and
     decoded (central differences, mean RGB over the picture);
  3. the bias that puts  latent @ J + b  on the decoded picture's mean;
  4. the tonal response r(v): the latent is stepped along pinv(J) for +-R and +-B by `unit` and the decoded
     change per unit, per brightness bin of the picture, is averaged — the decoder passes about 0.45 of a
     colour change near black and 1.4 in the midtones (SDXL).

Two or three pictures of different brightness give a usable average (the two measured on SDXL agreed to 4 %).
The entry is printed in the form axes.MEASURED takes (factors = 2J, bias = 2b - 1: the pack's unit range is
(projection + 1) / 2).

Usage (nothing in this process touches CUDA; the server does the decodes):
    python tools/calibrate_axes.py --ckpt "SDXL\\model.safetensors" --scale 0.13025 pic1.png pic2.png
    --shift for families whose latent format subtracts one (SD3: --scale 1.5305 --shift 0.0609;
    Flux: --scale 0.3611 --shift 0.1159). model space = (latent - shift) * scale.
"""

import argparse
import io
import json
import os
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_api import Server  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def run(server, graph):
    return server.wait(server.queue(graph))


def latent_file(history, node, output_dir):
    item = history["outputs"][node]["latents"][0]
    return os.path.join(output_dir, item.get("subfolder", ""), item["filename"])


def encode(server, ckpt, picture, output_dir):
    name = server.upload_image(picture, "colorkit_axes_" + os.path.basename(picture))
    g = {"ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
         "l": {"class_type": "LoadImage", "inputs": {"image": name}},
         "e": {"class_type": "VAEEncode", "inputs": {"pixels": ["l", 0], "vae": ["ck", 2]}},
         "s": {"class_type": "SaveLatent", "inputs": {"samples": ["e", 0], "filename_prefix": "colorkit_axes/latent"}},
         "d": {"class_type": "VAEDecode", "inputs": {"samples": ["e", 0], "vae": ["ck", 2]}},
         "p": {"class_type": "PreviewImage", "inputs": {"images": ["d", 0]}}}
    h = run(server, g)
    data = load_file(latent_file(h, "s", output_dir))
    decoded = server.images(h, "p")[0].astype(np.float64) / 255.0
    return data, decoded


def decode_many(server, ckpt, template, latents, input_dir):
    """latents: {key: VAE-space tensor} -> {key: decoded RGB 0..1 array}."""
    g = {"ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    for key, tensor in latents.items():
        out = dict(template)
        out["latent_tensor"] = tensor.float().contiguous()
        name = f"colorkit_axes_{key}.latent"
        save_file(out, os.path.join(input_dir, name))
        g[f"L{key}"] = {"class_type": "LoadLatent", "inputs": {"latent": name}}
        g[f"D{key}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"L{key}", 0], "vae": ["ck", 2]}}
        g[f"P{key}"] = {"class_type": "PreviewImage", "inputs": {"images": [f"D{key}", 0]}}
    h = run(server, g)
    return {key: server.images(h, f"P{key}")[0].astype(np.float64) / 255.0 for key in latents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pictures", nargs="+")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scale", type=float, required=True, help="latent_format.scale_factor")
    ap.add_argument("--shift", type=float, default=0.0, help="latent_format.shift_factor (SD3, Flux)")
    ap.add_argument("--delta", type=float, default=0.05, help="model-space nudge per channel")
    ap.add_argument("--unit", type=float, default=0.034, help="unit-range colour step for the tonal response")
    ap.add_argument("--comfy", required=True, help="ComfyUI root (its input/ and output/ folders)")
    ap.add_argument("--address", default="127.0.0.1:8188")
    args = ap.parse_args()
    server = Server(args.address)
    server.require()
    input_dir, output_dir = os.path.join(args.comfy, "input"), os.path.join(args.comfy, "output")

    def to_model(lat):
        return (lat - args.shift) * args.scale

    def to_vae(x):
        return x / args.scale + args.shift

    Js, biases, tone_rows = [], [], []
    bins = np.linspace(0.0, 1.0, 21)
    for picture in args.pictures:
        data, decoded = encode(server, args.ckpt, picture, output_dir)
        lat = data["latent_tensor"].float()
        x = to_model(lat)
        channels = x.shape[1]
        nudged = {}
        for c in range(channels):
            for sign, tag in ((1, "p"), (-1, "m")):
                y = x.clone()
                y[:, c] += sign * args.delta
                nudged[f"c{c}{tag}"] = to_vae(y)
        dec = decode_many(server, args.ckpt, data, nudged, input_dir)
        J = np.stack([(dec[f"c{c}p"].reshape(-1, 3).mean(0) - dec[f"c{c}m"].reshape(-1, 3).mean(0)) / (2 * args.delta)
                      for c in range(channels)])                                       # [C,3]
        Js.append(J)
        lm = x[0].numpy().reshape(channels, -1).T                                        # [N,C]
        biases.append(decoded.reshape(-1, 3).mean(0) - (lm @ J).mean(0))
        back = np.linalg.pinv(J)                                                         # [3,C]
        steps = {}
        for name, u in (("Rp", (args.unit, 0, 0)), ("Rm", (-args.unit, 0, 0)), ("Bp", (0, 0, args.unit)),
                        ("Bm", (0, 0, -args.unit))):
            dl = np.array(u) @ back
            y = x.clone()
            for c in range(channels):
                y[:, c] += float(dl[c])
            steps[name] = to_vae(y)
        dec = decode_many(server, args.ckpt, data, steps, input_dir)
        luma = decoded @ LUMA
        idx = np.digitize(luma, bins) - 1
        for name, ch, sign in (("Rp", 0, 1), ("Rm", 0, -1), ("Bp", 2, 1), ("Bm", 2, -1)):
            d = (dec[name] - decoded)[..., ch] * sign / args.unit
            tone_rows.append([d[idx == i].mean() if (idx == i).sum() > 2000 else np.nan for i in range(20)])
        print(f"{picture}: J condition {np.linalg.cond(J):.1f}; +R through pinv(J) decodes as "
              f"{(dec['Rp'] - decoded).reshape(-1, 3).mean(0).round(4)} for {args.unit} asked", file=sys.stderr)
    J = np.mean(Js, axis=0)
    b = np.mean(biases, axis=0)
    tone = np.nanmean(np.array(tone_rows), axis=0)
    # fill bins no picture reached from their neighbours
    for i in range(20):
        if not np.isfinite(tone[i]):
            near = [tone[j] for j in (i - 1, i + 1) if 0 <= j < 20 and np.isfinite(tone[j])]
            tone[i] = np.mean(near) if near else 1.0
    entry = {"factors": (2 * J).round(4).tolist(), "bias": (2 * b - 1).round(4).tolist(),
             "tone": {"v": [round(float((bins[i] + bins[i + 1]) / 2), 3) for i in range(20)],
                      "r": [round(float(v), 2) for v in tone]}}
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
