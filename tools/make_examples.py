"""Generate the example workflows in examples/ from the live node definitions.

Hand-written workflow JSON goes stale the moment a node gains or loses an
input. Building it from the definitions the running server publishes
(``GET /object_info``) means the examples always match the nodes that ship
beside them — and are read from the same source the editor builds its widgets
from, so what is written is what the editor expects. Start ComfyUI, then:

    python tools/make_examples.py [--server 127.0.0.1:8188]

The published examples leave the checkpoint empty. For a copy that opens ready
to queue, point --out at the ComfyUI user's workflow folder and name the model
exactly as the server lists it — ComfyUI builds those names with os.path.relpath,
so on Windows they carry backslashes, and a forward slash fails validation with
"Value not in list":

    python tools/make_examples.py --out <ComfyUI>/user/default/workflows/ColorGrade \\
        --checkpoint "SDXL\\protovisionXLHighFidelity3D_releaseV660Bakedvae.safetensors"
"""

import argparse
import json
import os
import sys
import urllib.request

# ComfyUI makes a widget of exactly these input types; every other type is a
# socket for a wire — the core ones and any a custom pack invents (IPADAPTER,
# CLIP_VISION, ...). The rule here used to be the other way round, a list of
# socket types with a widget for anything else, and the first foreign type an
# example used became a phantom widget that shifted every value after it.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


# The checkpoint the loaders start on. Left empty for the published examples,
# since nobody else has the same file; set with --checkpoint for a personal copy.
CHECKPOINT = None


def _checkpoint():
    # Always explicit: the server's dropdown lists THIS machine's files, and its
    # first entry is no default for anybody else.
    return {"ckpt_name": CHECKPOINT}


class ServerNode:
    """A node as the running server describes it, shaped like a node class."""

    def __init__(self, info):
        given = info.get("input", {})
        order = info.get("input_order") or {}
        self._inputs = {}
        for section in ("required", "optional"):
            entries = given.get(section) or {}
            # the editor lays widgets out in input_order; fall back to the dict's own order
            names = [n for n in order.get(section, []) if n in entries] or list(entries)
            self._inputs[section] = {n: entries[n] for n in names}
        self.RETURN_TYPES = tuple(info.get("output") or ())
        self.RETURN_NAMES = tuple(info.get("output_name") or self.RETURN_TYPES)

    def INPUT_TYPES(self):
        return self._inputs


class Workflow:
    def __init__(self):
        self.nodes = []
        self.links = []
        self._node_id = 0
        self._link_id = 0

    # -- building ---------------------------------------------------------- #

    def add(self, node_class, type_name, pos, values=None, size=(330, 120), title=None):
        self._node_id += 1
        spec = node_class.INPUT_TYPES()
        required = spec.get("required", {})
        optional = spec.get("optional", {})

        inputs, widgets, widget_names = [], [], []
        for name, definition in list(required.items()) + list(optional.items()):
            kind = definition[0]
            opts = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}

            # dropdowns come either as a bare list or as "COMBO" with its options
            choices = kind if isinstance(kind, (list, tuple)) else (
                opts.get("options") if kind == "COMBO" else None)
            forced = opts.get("forceInput")              # a widget type shown as a socket
            if choices is not None and not forced:
                inputs.append({"localized_name": name, "name": name, "type": "COMBO",
                               "widget": {"name": name}, "link": None})
                widget_names.append(name)
                widgets.append(opts.get("default", choices[0] if choices else None))
            elif isinstance(kind, str) and kind in WIDGET_TYPES and not forced:
                inputs.append({"localized_name": name, "name": name, "type": kind,
                               "widget": {"name": name}, "link": None})
                widget_names.append(name)
                widgets.append(opts.get("default", ""))
                if opts.get("control_after_generate"):
                    widget_names.append(f"{name}_control")
                    widgets.append("fixed")
            else:                                        # a socket, wired or left open
                entry = {"localized_name": name, "name": name, "type": str(kind), "link": None}
                if name in optional:
                    entry["shape"] = 7                   # the dashed "optional" dot
                inputs.append(entry)

        if values:
            for name, value in values.items():
                if name not in widget_names:
                    raise KeyError(f"{type_name} has no widget {name!r}; has {widget_names}")
                widgets[widget_names.index(name)] = value

        returns = getattr(node_class, "RETURN_TYPES", ())
        names = getattr(node_class, "RETURN_NAMES", returns)
        outputs = [{"localized_name": n, "name": n, "type": t, "links": []}
                   for t, n in zip(returns, names)]

        node = {
            "id": self._node_id, "type": type_name, "pos": list(pos),
            "size": list(size), "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": type_name},
            "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def link(self, src_node, src_slot, dst_node, dst_input_name):
        self._link_id += 1
        dst_slot = next(i for i, inp in enumerate(dst_node["inputs"])
                        if inp["name"] == dst_input_name)
        link_type = src_node["outputs"][src_slot]["type"]
        dst_node["inputs"][dst_slot]["link"] = self._link_id
        src_node["outputs"][src_slot]["links"].append(self._link_id)
        self.links.append([self._link_id, src_node["id"], src_slot,
                           dst_node["id"], dst_slot, link_type])

    def note(self, text, pos, size=(400, 180)):
        self._node_id += 1
        self.nodes.append({
            "id": self._node_id, "type": "MarkdownNote", "pos": list(pos),
            "size": list(size), "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": [], "outputs": [], "title": "Read me",
            "properties": {}, "widgets_values": [text],
            "color": "#432", "bgcolor": "#653",
        })

    def dump(self, path):
        data = {
            "id": os.path.splitext(os.path.basename(path))[0],
            "revision": 0,
            "last_node_id": self._node_id,
            "last_link_id": self._link_id,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        print(f"wrote {path}")


# --------------------------------------------------------------------------- #

def build_img2img(core, pack, out):
    """The image-space node cleaning up an img2img pass."""
    w = Workflow()
    w.note(
        "## A1111 Color Correction (img2img)\n\n"
        "An img2img pass drifts in colour, especially at higher denoise. This is "
        "the fix Automatic1111 applies: histogram-match the result to the input "
        "image in LAB, then keep the generated image's own brightness.\n\n"
        "**Wire the original image into `reference`.** Compare the two previews: "
        "the top one is raw, the bottom one corrected.\n\n"
        "`blend_mode = a1111` is the webui's exact behaviour. `lab_luminosity` is "
        "the opposite trade — it keeps the original colours and takes the corrected "
        "brightness.",
        (-480, 40), (420, 300))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    load = w.add(core["LoadImage"], "LoadImage", (20, 200), {"image": "example.png"}, size=(340, 320))
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 40),
                {"text": "a photograph, natural light, fine detail"}, (380, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 200),
                {"text": "blurry, low quality"}, (380, 120), "Negative")
    enc = w.add(core["VAEEncode"], "VAEEncode", (400, 360), size=(240, 60))
    ks = w.add(core["KSampler"], "KSampler", (820, 40),
               {"steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m",
                "scheduler": "karras", "denoise": 0.65}, (320, 280))
    dec = w.add(core["VAEDecode"], "VAEDecode", (1180, 40), size=(240, 60))
    raw = w.add(core["PreviewImage"], "PreviewImage", (1180, 140), size=(320, 300),
                title="Before (drifted)")
    cc = w.add(pack["A1111ColorCorrection"], "A1111ColorCorrection", (1480, 40),
               {"strength": 1.0, "blend_mode": "a1111"}, (330, 180))
    out_img = w.add(core["SaveImage"], "SaveImage", (1850, 40), size=(380, 400),
                    title="After (corrected)")

    w.link(ckpt, 0, ks, "model")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(load, 0, enc, "pixels")
    w.link(ckpt, 2, enc, "vae")
    w.link(enc, 0, ks, "latent_image")
    w.link(pos, 0, ks, "positive")
    w.link(neg, 0, ks, "negative")
    w.link(ks, 0, dec, "samples")
    w.link(ckpt, 2, dec, "vae")
    w.link(dec, 0, raw, "images")
    w.link(dec, 0, cc, "image")
    w.link(load, 0, cc, "reference")
    w.link(cc, 0, out_img, "images")
    w.dump(os.path.join(out, "01_img2img_color_correction.json"))


def build_guidance(core, pack, out):
    """The model patch holding colour during sampling."""
    w = Workflow()
    w.note(
        "## Latent Color Guidance\n\n"
        "Instead of repairing the colours afterwards, hold them during sampling. "
        "The patch corrects the model's own prediction at every step, so the image "
        "is composed in the right palette rather than pushed into it.\n\n"
        "It patches `MODEL`, so **any** sampler works downstream — KSampler, "
        "KSamplerAdvanced, SamplerCustom, detailers.\n\n"
        "- `strength` 0.4-0.7 is the usual range.\n"
        "- `start_percent` 0.15 skips the first steps, where the prediction is "
        "still featureless and its spread means nothing.\n"
        "- `space = auto` projects through the model's own latent-to-RGB matrix, "
        "so this works on SD1.5, SDXL, SD3, Flux and the video models alike.\n\n"
        "Bypass the patch node (Ctrl+B) to see the difference.",
        (-480, 40), (420, 380))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    load = w.add(core["LoadImage"], "LoadImage", (20, 200), {"image": "example.png"}, size=(340, 320))
    enc = w.add(core["VAEEncode"], "VAEEncode", (400, 560), size=(240, 60),
                title="Reference latent")
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 40),
                {"text": "a photograph, natural light, fine detail"}, (380, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 200),
                {"text": "blurry, low quality"}, (380, 120), "Negative")
    guide = w.add(pack["ApplyLatentColorGuidance"], "ApplyLatentColorGuidance", (820, 400),
                  {"strength": 0.6, "start_percent": 0.15, "end_percent": 0.9}, (340, 240))
    ks = w.add(core["KSampler"], "KSampler", (1200, 40),
               {"steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m",
                "scheduler": "karras", "denoise": 0.75}, (320, 280))
    dec = w.add(core["VAEDecode"], "VAEDecode", (1560, 40), size=(240, 60))
    save = w.add(core["SaveImage"], "SaveImage", (1560, 140), size=(380, 400))

    w.link(ckpt, 0, guide, "model")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(load, 0, enc, "pixels")
    w.link(ckpt, 2, enc, "vae")
    w.link(enc, 0, guide, "reference_latent")
    w.link(enc, 0, ks, "latent_image")
    w.link(guide, 0, ks, "model")
    w.link(pos, 0, ks, "positive")
    w.link(neg, 0, ks, "negative")
    w.link(ks, 0, dec, "samples")
    w.link(ckpt, 2, dec, "vae")
    w.link(dec, 0, save, "images")
    w.dump(os.path.join(out, "02_latent_color_guidance.json"))


def build_both(core, pack, out):
    """Guidance during sampling plus the image node as a final clean-up."""
    w = Workflow()
    w.note(
        "## Both stages together\n\n"
        "The usual production setup for an upscale or refine pass that has to keep "
        "the original palette:\n\n"
        "1. **Latent Color Guidance** holds the colour while the model works, "
        "so it never commits to the wrong palette.\n"
        "2. **A1111 Color Correction** cleans up whatever drift is left after the "
        "decode, which is cheap and exact.\n\n"
        "`Latent Color Match` is also wired here, running on the init latent before "
        "sampling starts — useful when the input itself is already off.",
        (-480, 40), (420, 320))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    load = w.add(core["LoadImage"], "LoadImage", (20, 200), {"image": "example.png"}, size=(340, 320))
    enc = w.add(core["VAEEncode"], "VAEEncode", (400, 560), size=(240, 60),
                title="Reference latent")
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 40),
                {"text": "a photograph, natural light, fine detail"}, (380, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 200),
                {"text": "blurry, low quality"}, (380, 120), "Negative")
    match = w.add(pack["LatentColorMatch"], "LatentColorMatch", (820, 640),
                  {"strength": 1.0}, (330, 200), "Optional: fix the init latent")
    guide = w.add(pack["ApplyLatentColorGuidance"], "ApplyLatentColorGuidance", (820, 360),
                  {"strength": 0.5}, (340, 240))
    ks = w.add(core["KSampler"], "KSampler", (1200, 40),
               {"steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m",
                "scheduler": "karras", "denoise": 0.7}, (320, 280))
    dec = w.add(core["VAEDecode"], "VAEDecode", (1560, 40), size=(240, 60))
    cc = w.add(pack["A1111ColorCorrection"], "A1111ColorCorrection", (1560, 140),
               {"strength": 0.8, "blend_mode": "a1111"}, (330, 180))
    save = w.add(core["SaveImage"], "SaveImage", (1930, 40), size=(380, 400))

    w.link(ckpt, 0, guide, "model")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(load, 0, enc, "pixels")
    w.link(ckpt, 2, enc, "vae")
    w.link(enc, 0, guide, "reference_latent")
    w.link(enc, 0, match, "samples")
    w.link(enc, 0, match, "reference")
    w.link(ckpt, 0, match, "model")
    w.link(match, 0, ks, "latent_image")
    w.link(guide, 0, ks, "model")
    w.link(pos, 0, ks, "positive")
    w.link(neg, 0, ks, "negative")
    w.link(ks, 0, dec, "samples")
    w.link(ckpt, 2, dec, "vae")
    w.link(dec, 0, cc, "image")
    w.link(load, 0, cc, "reference")
    w.link(cc, 0, save, "images")
    w.dump(os.path.join(out, "03_guidance_plus_correction.json"))


def build_loopback(core, pack, out):
    """A1111's Loopback: a chain of img2img passes anchored to one reference."""
    w = Workflow()
    w.note(
        "## A1111 Loopback\n\n"
        "This is what people remember as A1111's colour correction being \"built "
        "into sampling\". It is not: the webui has no correction inside its "
        "sampler. It is the Loopback script.\n\n"
        "Each pass is a full img2img, so the **content** is reworked every time — "
        "that is why it looks like more than a colour filter. What stays fixed is "
        "the reference: measured once from the starting image and reused by every "
        "pass, so the palette cannot drift however long the chain runs.\n\n"
        "- `denoise` falls from the first pass to `final_denoise`, so the chain "
        "settles instead of wandering.\n"
        "- `hold_exposure` is ours, not A1111's. Its blend takes lightness from "
        "the generated image, so a long chain holds colour but can still wander "
        "brighter or darker. Raise this if yours does.\n"
        "- Wire a `mask` to rework only part of the frame: everything outside is "
        "pasted back from the original after each pass, so it does not soften.\n\n"
        "`all_passes` is every pass as a batch — scrub it to see the progression.",
        (-500, 40), (440, 420))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    load = w.add(core["LoadImage"], "LoadImage", (20, 200), {"image": "example.png"}, size=(340, 320))
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 40),
                {"text": "a photograph, natural light, fine detail"}, (380, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 200),
                {"text": "blurry, low quality"}, (380, 120), "Negative")
    loop = w.add(pack["A1111Loopback"], "A1111Loopback", (820, 40),
                 {"loops": 4, "steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m",
                  "scheduler": "karras", "denoise": 0.75, "final_denoise": 0.5,
                  "denoise_curve": "Linear", "color_correction": True},
                 (360, 460))
    every = w.add(core["PreviewImage"], "PreviewImage", (1220, 40), size=(340, 320),
                  title="Every pass")
    final = w.add(core["SaveImage"], "SaveImage", (1220, 400), size=(380, 400),
                  title="Final pass")

    w.link(ckpt, 0, loop, "model")
    w.link(ckpt, 2, loop, "vae")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(pos, 0, loop, "positive")
    w.link(neg, 0, loop, "negative")
    w.link(load, 0, loop, "image")
    w.link(loop, 0, every, "images")
    w.link(loop, 1, final, "images")
    w.dump(os.path.join(out, "04_a1111_loopback.json"))


def build_grade_image(core, pack, out):
    """The quickest way to feel the wheels: no model, no sampling."""
    w = Workflow()
    w.note(
        "## Start here\n\n"
        "Load a picture, turn the wheels, queue. Nothing is sampled and no "
        "checkpoint is loaded, so the result comes back immediately — this is "
        "the fastest way to learn what each wheel does.\n\n"
        "**Lift** moves the shadows and leaves white alone. **Gain** moves the "
        "highlights and leaves black alone. **Gamma** bends everything in "
        "between. The bar under each disc is that range's brightness.\n\n"
        "Drag the handle — it moves *with* the pointer rather than jumping to "
        "it, so small corrections are possible. Hold **Shift** for fine "
        "control, **Alt-click** to reset one wheel.\n\n"
        "Right-click the node for presets, and for saving the current settings "
        "as one of your own.",
        (-520, 40), (460, 400))

    load = w.add(core["LoadImage"], "LoadImage", (20, 40), {"image": "example.png"}, size=(340, 320))
    # heights as the 1.5x frontend lays the wheels out (measured 30.10: about 396 with the numbers hidden)
    grade = w.add(pack["ImageColorGrade"], "ImageColorGrade", (400, 40), size=(640, 420))
    before = w.add(core["PreviewImage"], "PreviewImage", (1080, 40), size=(320, 300),
                   title="Before")
    after = w.add(core["SaveImage"], "SaveImage", (1080, 420), size=(380, 400),
                  title="After")

    w.link(load, 0, grade, "image")
    w.link(load, 0, before, "images")
    w.link(grade, 0, after, "images")
    w.dump(os.path.join(out, "05_grade_an_image.json"))


def build_grade_latent(core, pack, out):
    """Grading while the image is made, with the post-pass alongside."""
    w = Workflow()
    w.note(
        "## Grading during sampling\n\n"
        "The wheels on the **latent** node change the model's own prediction at "
        "every step, so the picture is *composed* in the graded look instead of "
        "having it laid over the top. Shadows get the colour you asked for "
        "because the model put it there, not because a filter pushed it.\n\n"
        "No input image is needed — this generates from the prompt, so it runs "
        "straight away.\n\n"
        "The lower sampler renders the same seed on the **ungraded** model, and "
        "the image node grades that render afterwards. Set the same wheels on both "
        "grade nodes to see how differently a grade lands when the model composes "
        "in it and when it is laid over a finished picture.\n\n"
        "`strength` scales the whole grade; `start_percent` / `end_percent` "
        "limit it to part of the schedule.",
        (-520, 40), (460, 430))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    empty = w.add(core["EmptyLatentImage"], "EmptyLatentImage", (20, 180),
                  {"width": 1024, "height": 1024}, (340, 110))
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (20, 330),
                {"text": "a quiet street at dusk, wet pavement, shop windows, photograph"},
                (340, 140), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (20, 500),
                {"text": "blurry, low quality"}, (340, 110), "Negative")

    # the latent wheels are taller than the image wheels (strength and the window on top); the samplers sit
    # below them and the decoders right of the samplers, clear of the post-pass grade node (30.10: overlapping)
    grade = w.add(pack["LatentColorGrade"], "LatentColorGrade", (400, 40), size=(640, 470))
    ks = w.add(core["KSampler"], "KSampler", (400, 570),
               {"steps": 26, "cfg": 6.5, "sampler_name": "dpmpp_2m",
                "scheduler": "karras", "denoise": 1.0}, (320, 280))
    dec = w.add(core["VAEDecode"], "VAEDecode", (760, 570), size=(240, 60))
    save = w.add(core["SaveImage"], "SaveImage", (1080, 570), size=(380, 380),
                 title="Graded while sampling")

    # The post-pass needs a render that was NOT graded while sampling. It used to
    # take the graded decode, so with any real grade it showed the wheels applied
    # twice — invisible only while every wheel sat at its neutral default.
    ks_plain = w.add(core["KSampler"], "KSampler", (400, 900),
                     {"steps": 26, "cfg": 6.5, "sampler_name": "dpmpp_2m",
                      "scheduler": "karras", "denoise": 1.0}, (320, 280),
                     "Same seed, ungraded model")
    dec_plain = w.add(core["VAEDecode"], "VAEDecode", (760, 900), size=(240, 60))
    plain_out = w.add(core["PreviewImage"], "PreviewImage", (760, 1000), size=(300, 300),
                      title="No grade")
    post = w.add(pack["ImageColorGrade"], "ImageColorGrade", (1080, 40), size=(640, 420),
                 title="The same wheels, applied afterwards")
    post_out = w.add(core["PreviewImage"], "PreviewImage", (1760, 40), size=(320, 300),
                     title="Graded afterwards")

    w.link(ckpt, 0, grade, "model")
    w.link(grade, 0, ks, "model")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(pos, 0, ks, "positive")
    w.link(neg, 0, ks, "negative")
    w.link(empty, 0, ks, "latent_image")
    w.link(ks, 0, dec, "samples")
    w.link(ckpt, 2, dec, "vae")
    w.link(dec, 0, save, "images")

    w.link(ckpt, 0, ks_plain, "model")
    w.link(pos, 0, ks_plain, "positive")
    w.link(neg, 0, ks_plain, "negative")
    w.link(empty, 0, ks_plain, "latent_image")
    w.link(ks_plain, 0, dec_plain, "samples")
    w.link(ckpt, 2, dec_plain, "vae")
    w.link(dec_plain, 0, plain_out, "images")
    w.link(dec_plain, 0, post, "image")
    w.link(post, 0, post_out, "images")
    w.dump(os.path.join(out, "06_grade_while_sampling.json"))


def build_ipadapter(core, pack, out):
    """IP-Adapter on its own, and IP-Adapter with the colour pinned.

    The two transfer different things from the same picture, which is why
    they belong together rather than in competition.
    """
    missing = [n for n in ("IPAdapterUnifiedLoader", "IPAdapterAdvanced") if n not in core]
    if missing:
        print(f"  skipping the IP-Adapter example: {', '.join(missing)} not installed")
        return

    w = Workflow()
    w.note(
        "## IP-Adapter, and IP-Adapter with the colour held\n\n"
        "Both take something from the same reference picture, but not the same "
        "thing. IP-Adapter encodes it with CLIP Vision and feeds the result into "
        "the model's attention, which carries **subject, style and texture**. It "
        "is famously loose about the palette: the look arrives, the exact colours "
        "wander.\n\n"
        "The Latent Color Guidance node measures the reference's **colour** and holds the "
        "prediction to it. Different hook, different quantity, no conflict — one "
        "patches attention, the other patches the noise prediction.\n\n"
        "Both branches run the **same seed**, so the only difference is the "
        "colour node. The comparison at the end is the point of the graph.\n\n"
        "**Before running:** pick an SDXL checkpoint, and make sure the "
        "IP-Adapter models and CLIP Vision are in place — the unified loader "
        "will say so if they are not.",
        (-540, 40), (480, 440))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    load = w.add(core["LoadImage"], "LoadImage", (20, 190), {"image": "example.png"}, size=(340, 320),
                 title="The reference — style and colour both come from here")
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (20, 540),
                {"text": "a harbour at dawn, boats, wet stone, photograph"},
                (340, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (20, 700),
                {"text": "blurry, low quality"}, (340, 110), "Negative")
    empty = w.add(core["EmptyLatentImage"], "EmptyLatentImage", (20, 850),
                  {"width": 1024, "height": 1024}, (340, 110))

    loader = w.add(core["IPAdapterUnifiedLoader"], "IPAdapterUnifiedLoader", (420, 40),
                   {"preset": "PLUS (high strength)"}, (340, 100))
    ip = w.add(core["IPAdapterAdvanced"], "IPAdapterAdvanced", (420, 190),
               {"weight": 0.8, "start_at": 0.0, "end_at": 1.0}, (340, 220))

    encode = w.add(core["VAEEncode"], "VAEEncode", (420, 460), size=(260, 60),
                   title="Reference, as a latent")
    guide = w.add(pack["ApplyLatentColorGuidance"], "ApplyLatentColorGuidance", (420, 560),
                  {"strength": 0.6, "start_percent": 0.15, "end_percent": 0.9},
                  (340, 240), "The colour, held")

    seed = 777000
    plain = w.add(core["KSampler"], "KSampler", (820, 40),
                  {"seed": seed, "steps": 28, "cfg": 6.0, "sampler_name": "dpmpp_2m",
                   "scheduler": "karras", "denoise": 1.0}, (320, 280),
                  "IP-Adapter only")
    held = w.add(core["KSampler"], "KSampler", (820, 360),
                 {"seed": seed, "steps": 28, "cfg": 6.0, "sampler_name": "dpmpp_2m",
                  "scheduler": "karras", "denoise": 1.0}, (320, 280),
                 "IP-Adapter + Color Grade")

    decode_a = w.add(core["VAEDecode"], "VAEDecode", (1180, 40), size=(240, 60))
    decode_b = w.add(core["VAEDecode"], "VAEDecode", (1180, 140), size=(240, 60))
    preview_a = w.add(core["PreviewImage"], "PreviewImage", (1180, 240), size=(300, 280),
                      title="IP-Adapter only")
    preview_b = w.add(core["PreviewImage"], "PreviewImage", (1180, 560), size=(300, 280),
                      title="with the colour held")

    compare = w.add(pack["ColorKitCompare"], "ColorKitCompare", (1520, 40),
                    {"mode": "wipe", "position": 0.5}, (320, 180))
    shown = w.add(core["SaveImage"], "SaveImage", (1520, 260), size=(360, 380),
                  title="Side by side")
    scopes = w.add(pack["ColorKitScopes"], "ColorKitScopes", (1520, 680),
                   {"mode": "all", "size": 360}, (320, 140))
    scope_out = w.add(core["PreviewImage"], "PreviewImage", (1880, 680), size=(360, 220),
                      title="Scopes of the held version")

    w.link(ckpt, 0, loader, "model")
    w.link(loader, 0, ip, "model")
    w.link(loader, 1, ip, "ipadapter")
    w.link(load, 0, ip, "image")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")

    w.link(load, 0, encode, "pixels")
    w.link(ckpt, 2, encode, "vae")
    w.link(ip, 0, guide, "model")
    w.link(encode, 0, guide, "reference_latent")

    for sampler, model_node in ((plain, ip), (held, guide)):
        w.link(model_node, 0, sampler, "model")
        w.link(pos, 0, sampler, "positive")
        w.link(neg, 0, sampler, "negative")
        w.link(empty, 0, sampler, "latent_image")

    w.link(plain, 0, decode_a, "samples")
    w.link(held, 0, decode_b, "samples")
    w.link(ckpt, 2, decode_a, "vae")
    w.link(ckpt, 2, decode_b, "vae")
    w.link(decode_a, 0, preview_a, "images")
    w.link(decode_b, 0, preview_b, "images")

    w.link(decode_a, 0, compare, "image_a")
    w.link(decode_b, 0, compare, "image_b")
    w.link(compare, 0, shown, "images")
    w.link(decode_b, 0, scopes, "image")
    w.link(scopes, 0, scope_out, "images")

    w.dump(os.path.join(out, "07_ipadapter_with_color_grade.json"))


def build_palette(core, pack, out):
    """Two pictures in: the model redraws the first in the second's palette."""
    w = Workflow()
    w.note(
        "## A palette from any picture\n\n"
        "Two pictures in: **yours** (top) and a **colour source** (bottom) — a film "
        "still, a painting, a photo whose light you like. The model redraws yours in "
        "the source's palette:\n\n"
        "1. **Latent Color Match** moves the starting latent towards the source's "
        "colours.\n"
        "2. **Latent Color Guidance** holds that palette at every step while the "
        "model draws, so light and shadow are composed in it.\n"
        "3. **A1111 Color Correction** finishes on the decoded picture: colour from "
        "the source, brightness from the render.\n\n"
        "`denoise` 0.5 keeps your composition; raise it for a freer redraw. To try "
        "another look, change only the bottom picture.",
        (-520, 40), (460, 470))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    picture = w.add(core["LoadImage"], "LoadImage", (20, 200), {"image": "example.png"},
                    size=(340, 330), title="Your picture")
    source = w.add(core["LoadImage"], "LoadImage", (20, 590), {"image": "example.png"},
                   size=(340, 330), title="Colour source — any picture")
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 40),
                {"text": "describe your picture here"}, (380, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 200),
                {"text": "blurry, low quality"}, (380, 120), "Negative")
    enc_pic = w.add(core["VAEEncode"], "VAEEncode", (400, 380), size=(240, 60),
                    title="Your picture → latent")
    enc_src = w.add(core["VAEEncode"], "VAEEncode", (400, 590), size=(240, 60),
                    title="Colour source → latent")
    guide = w.add(pack["ApplyLatentColorGuidance"], "ApplyLatentColorGuidance", (820, 40),
                  {"strength": 0.5}, (340, 260))
    match = w.add(pack["LatentColorMatch"], "LatentColorMatch", (820, 380),
                  {"strength": 1.0}, (340, 200))
    ks = w.add(core["KSampler"], "KSampler", (1200, 40),
               {"steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m",
                "scheduler": "karras", "denoise": 0.5}, (320, 280))
    dec = w.add(core["VAEDecode"], "VAEDecode", (1560, 40), size=(240, 60))
    cc = w.add(pack["A1111ColorCorrection"], "A1111ColorCorrection", (1560, 140),
               {"strength": 0.8, "blend_mode": "a1111"}, (330, 180))
    save = w.add(core["SaveImage"], "SaveImage", (1930, 40), size=(400, 440),
                 title="In the source's palette")

    w.link(ckpt, 0, guide, "model")
    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(picture, 0, enc_pic, "pixels")
    w.link(source, 0, enc_src, "pixels")
    w.link(ckpt, 2, enc_pic, "vae")
    w.link(ckpt, 2, enc_src, "vae")
    w.link(enc_src, 0, guide, "reference_latent")
    w.link(enc_pic, 0, match, "samples")
    w.link(enc_src, 0, match, "reference")
    w.link(ckpt, 0, match, "model")
    w.link(match, 0, ks, "latent_image")
    w.link(guide, 0, ks, "model")
    w.link(pos, 0, ks, "positive")
    w.link(neg, 0, ks, "negative")
    w.link(ks, 0, dec, "samples")
    w.link(ckpt, 2, dec, "vae")
    w.link(dec, 0, cc, "image")
    w.link(source, 0, cc, "reference")
    w.link(cc, 0, save, "images")
    w.dump(os.path.join(out, "08_palette_from_any_picture.json"))


# the circles template 09 ships with: shadows a little towards teal, highlights towards warm, a touch more
# saturation and contrast — moved, so the first run already shows what they do
WARM_SHIFT = {"lift_r": -0.03, "lift_b": 0.06, "gain_r": 0.09, "gain_g": 0.03, "gain_b": -0.08,
              "saturation": 1.15, "contrast": 1.05}


def build_palette_wheels(core, pack, out):
    """A picture's palette from Latent Color Match, the latent wheels on top, the palette alone beside it."""
    w = Workflow()
    w.note(
        "## A picture's palette + your own wheels\n\n"
        "Two pictures in, as in template 08: **yours** (top) and a **colour source** "
        "(bottom). **Latent Color Match** starts the run in the source's palette; "
        "**Grade Wheels (latent)** takes its bearings from the model's first "
        "prediction — already in that palette — holds it while the model draws, and "
        "adds your adjustment on top. The circles here push the shadows towards teal "
        "and the highlights towards warm.\n\n"
        "The lower sampler renders the same seed with the palette alone, so the two "
        "outputs show what the wheels add. Move a circle and run again: drag, Shift "
        "for fine moves, Alt-click to reset, right-click for presets.\n\n"
        "`denoise` 0.5 keeps your composition. Leave Latent Color Guidance and A1111 "
        "Color Correction out of this chain: both pull back towards the source and "
        "would take your adjustment with them.",
        (-520, 40), (460, 520))

    ckpt = w.add(core["CheckpointLoaderSimple"], "CheckpointLoaderSimple", (20, 40),
                 values=_checkpoint(), size=(340, 100))
    picture = w.add(core["LoadImage"], "LoadImage", (20, 200), {"image": "example.png"},
                    size=(340, 330), title="Your picture")
    source = w.add(core["LoadImage"], "LoadImage", (20, 590), {"image": "example.png"},
                   size=(340, 330), title="Colour source — any picture")
    pos = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 40),
                {"text": "describe your picture here"}, (380, 120), "Positive")
    neg = w.add(core["CLIPTextEncode"], "CLIPTextEncode", (400, 200),
                {"text": "blurry, low quality"}, (380, 120), "Negative")
    enc_pic = w.add(core["VAEEncode"], "VAEEncode", (400, 380), size=(240, 60),
                    title="Your picture → latent")
    enc_src = w.add(core["VAEEncode"], "VAEEncode", (400, 590), size=(240, 60),
                    title="Colour source → latent")
    match = w.add(pack["LatentColorMatch"], "LatentColorMatch", (400, 700),
                  {"strength": 1.0}, (340, 200))
    grade = w.add(pack["LatentColorGrade"], "LatentColorGrade", (820, 40), WARM_SHIFT,
                  size=(640, 470), title="Your wheels, inside sampling")
    sampler = {"steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
               "denoise": 0.5}
    ks = w.add(core["KSampler"], "KSampler", (820, 570), sampler, (320, 280),
               "With your wheels")
    ks_plain = w.add(core["KSampler"], "KSampler", (820, 900), sampler, (320, 280),
                     "Palette only, same seed")
    dec = w.add(core["VAEDecode"], "VAEDecode", (1180, 570), size=(240, 60))
    dec_plain = w.add(core["VAEDecode"], "VAEDecode", (1180, 900), size=(240, 60))
    save = w.add(core["SaveImage"], "SaveImage", (1500, 40), size=(420, 460),
                 title="Palette + your wheels")
    plain_out = w.add(core["PreviewImage"], "PreviewImage", (1500, 560), size=(420, 460),
                      title="Palette only")

    w.link(ckpt, 1, pos, "clip")
    w.link(ckpt, 1, neg, "clip")
    w.link(picture, 0, enc_pic, "pixels")
    w.link(source, 0, enc_src, "pixels")
    w.link(ckpt, 2, enc_pic, "vae")
    w.link(ckpt, 2, enc_src, "vae")
    w.link(enc_pic, 0, match, "samples")
    w.link(enc_src, 0, match, "reference")
    w.link(ckpt, 0, match, "model")
    w.link(ckpt, 0, grade, "model")
    for k in (ks, ks_plain):
        w.link(match, 0, k, "latent_image")
        w.link(pos, 0, k, "positive")
        w.link(neg, 0, k, "negative")
    w.link(grade, 0, ks, "model")
    w.link(ckpt, 0, ks_plain, "model")
    w.link(ks, 0, dec, "samples")
    w.link(ks_plain, 0, dec_plain, "samples")
    w.link(ckpt, 2, dec, "vae")
    w.link(ckpt, 2, dec_plain, "vae")
    w.link(dec, 0, save, "images")
    w.link(dec_plain, 0, plain_out, "images")
    w.dump(os.path.join(out, "09_palette_plus_your_wheels.json"))


# what the nine examples cannot be built without
NEEDED = ("CheckpointLoaderSimple", "LoadImage", "KSampler", "ImageColorGrade",
          "LatentColorGrade", "ApplyLatentColorGuidance", "IPAdapterUnifiedLoader",
          "IPAdapterAdvanced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:8188",
                    help="a running ComfyUI with this pack and IPAdapter Plus installed")
    ap.add_argument("--out", default="")
    ap.add_argument("--checkpoint", default=None,
                    help="ckpt_name for every checkpoint loader, exactly as the server "
                         "lists it (on Windows with backslashes, e.g. SDXL\\model.safetensors)")
    args = ap.parse_args()

    global CHECKPOINT
    CHECKPOINT = args.checkpoint

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(here, "examples")

    try:
        with urllib.request.urlopen(f"http://{args.server}/object_info", timeout=60) as resp:
            info = json.load(resp)
    except OSError as exc:
        sys.exit(f"no ComfyUI answering at {args.server} ({exc}) — start it, or pass --server")

    nodes = {name: ServerNode(definition) for name, definition in info.items()}
    missing = [name for name in NEEDED if name not in nodes]
    if missing:
        sys.exit(f"the server at {args.server} has no {missing} — "
                 "are this pack and IPAdapter Plus installed in that ComfyUI?")
    os.makedirs(out, exist_ok=True)

    for build in (build_img2img, build_guidance, build_both, build_loopback,
                  build_grade_image, build_grade_latent, build_ipadapter, build_palette,
                  build_palette_wheels):
        build(nodes, nodes, out)


if __name__ == "__main__":
    main()
