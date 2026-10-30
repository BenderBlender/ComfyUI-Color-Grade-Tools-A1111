"""Talk to the ComfyUI that is already running, instead of importing it.

The demonstration scripts used to import ComfyUI as a library and call the
node classes directly. That skips everything ``main.py`` and ``execution.py``
put around a prompt — the ``torch.inference_mode()`` guard that every node
relies on, the allocator, DynamicVRAM, the queue, the Interrupt button — and
the pictures were made in a process the user could neither watch nor stop.
Going through the server the user already has open gives all of that back,
and nothing in this process touches CUDA.

The protocol is the one ComfyUI ships in
``script_examples/websockets_api_example.py``: POST ``/prompt``, wait, read
``/history/{id}``, fetch the pictures with ``/view``. Completion is polled
from ``/history`` rather than taken from the websocket, so the standard
library is enough; the embedded Python that comes with ComfyUI has no
``websocket-client`` and this is one dependency fewer for a published repo.

Three layers, each usable on its own:

* :class:`Server` — the HTTP calls, plus :meth:`Server.validate`, which
  checks a graph against ``/object_info`` (every class, every input name,
  every combo value, every link) before it is queued.
* ``graph_txt2img`` / ``graph_img2img`` / ``graph_lut`` — the graphs the
  scripts need, as plain dicts in the API's prompt format.
* :class:`Session` and :class:`Generator` — queue one generation, wait for
  it, read the picture back, rest for ``gap`` seconds. In dry-run mode the
  same calls validate, print, and return a placeholder instead of queueing.
"""

import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np

DEFAULT_ADDRESS = "127.0.0.1:8188"
POLL_SECONDS = 1.0
OUTPUT_NODE = "preview"             # the node whose image a graph hands back

CORE_CLASSES = ("CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage",
                "LoadImage", "VAEEncode", "VAEDecode", "KSampler", "PreviewImage")
PACK_CLASSES = ("ApplyLatentColorGuidance", "LatentColorGrade", "ColorKitApplyLUT")

# Inputs whose combo list is a snapshot of a folder; the file may be uploaded
# after validation, and the server checks it again when the prompt is queued.
FOLDER_COMBOS = {("LoadImage", "image")}


class ServerError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #

class Server:
    def __init__(self, address=DEFAULT_ADDRESS, timeout=60.0):
        if "://" in address:
            address = address.split("://", 1)[1]
        self.address = address.strip().rstrip("/")
        self.timeout = timeout
        self.reachable = None
        self._info = {}

    # -- plumbing --------------------------------------------------------- #

    def url(self, path, **query):
        url = f"http://{self.address}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _request(self, path, data=None, content_type=None, query=None):
        request = urllib.request.Request(self.url(path, **(query or {})), data=data)
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ServerError(f"{exc.code} from {path}: {_pretty_error(body)}") from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ServerError(f"no ComfyUI answering at http://{self.address} ({reason})") from None

    def get_json(self, path, **query):
        return json.loads(self._request(path, query=query) or b"{}")

    def post_json(self, path, payload):
        return json.loads(self._request(path, data=json.dumps(payload).encode("utf-8"),
                                        content_type="application/json") or b"{}")

    def unreachable_message(self):
        return (f"no ComfyUI answering at http://{self.address}. Start it the usual way "
                f"(Start ComfyUI.bat) and run this again, or pass --server host:port.")

    def ping(self):
        try:
            self.stats()
            self.reachable = True
        except ServerError:
            self.reachable = False
        return self.reachable

    def require(self):
        if not self.ping():
            raise ServerError(self.unreachable_message())

    # -- reading ---------------------------------------------------------- #

    def stats(self):
        return self.get_json("/system_stats")

    def memory_line(self):
        """What the server reports, so a run says what it is costing."""
        try:
            stats = self.stats()
        except ServerError as exc:
            return f"    [server] {exc}"
        gb = 1024 ** 3
        system = stats.get("system", {})
        device = (stats.get("devices") or [{}])[0]
        parts = []
        if device.get("vram_total"):
            used = device["vram_total"] - device.get("vram_free", 0)
            parts.append(f"VRAM {used / gb:.1f}/{device['vram_total'] / gb:.1f} GB")
        if system.get("ram_total"):
            used = system["ram_total"] - system.get("ram_free", 0)
            parts.append(f"RAM {used / gb:.0f}/{system['ram_total'] / gb:.0f} GB")
        return "    [server] " + "  ".join(parts)

    def object_info(self, cls=None):
        """``/object_info`` for one class (None when the server has no such
        node), or the whole table."""
        if cls is None:
            data = self.get_json("/object_info")
            self._info.update(data)
            return data
        if cls not in self._info:
            data = self.get_json(f"/object_info/{urllib.parse.quote(cls)}")
            self._info[cls] = data.get(cls)
        return self._info[cls]

    def combo_options(self, cls, name):
        info = self.object_info(cls)
        if info is None:
            raise ServerError(f"the server has no node class {cls!r}")
        spec = _input_specs(info).get(name)
        if spec is None:
            raise ServerError(f"{cls} has no input {name!r}")
        options = _combo_options(spec)
        if options is None:
            raise ServerError(f"{cls}.{name} is not a list input")
        return options

    def checkpoints(self):
        return self.combo_options("CheckpointLoaderSimple", "ckpt_name")

    def resolve_checkpoint(self, given):
        """A name the server's Load Checkpoint node lists. A bare file name or
        a full path is matched by its base name."""
        names = self.checkpoints()
        if given in names:
            return given
        wanted = os.path.basename(given.replace("\\", "/")).lower()
        hits = [n for n in names if os.path.basename(n.replace("\\", "/")).lower() == wanted]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise ServerError(f"{given!r} matches several checkpoints the server lists "
                              f"({hits}); pass one of them in full")
        stem = os.path.splitext(wanted)[0][:8]
        near = [n for n in names if stem and stem in n.lower()][:8]
        raise ServerError(
            f"the server's Load Checkpoint node does not list {given!r} "
            f"({len(names)} checkpoints known" + (f"; similar: {near}" if near else "") + "). "
            "Put the file where that ComfyUI looks (models/checkpoints, or a folder in its "
            "extra_model_paths.yaml) — a path on this machine is not enough.")

    # -- writing ---------------------------------------------------------- #

    def upload_image(self, data, name=None, overwrite=True):
        """PNG bytes (or a path) into the server's input folder; returns the
        name a LoadImage node takes."""
        if isinstance(data, (str, os.PathLike)):
            name = name or os.path.basename(data)
            with open(data, "rb") as fh:
                data = fh.read()
        if not name:
            raise ValueError("upload_image needs a file name for in-memory data")
        body, content_type = _multipart(
            {"overwrite": "true" if overwrite else "false", "type": "input"},
            {"image": (name, data, "image/png")})
        response = json.loads(self._request("/upload/image", data=body, content_type=content_type))
        subfolder = response.get("subfolder") or ""
        return f"{subfolder}/{response['name']}" if subfolder else response["name"]

    def queue(self, graph):
        response = self.post_json("/prompt", {"prompt": graph})
        if response.get("node_errors"):
            raise ServerError("the server rejected the graph: "
                              + json.dumps(response["node_errors"], indent=2))
        return response["prompt_id"]

    def history(self, prompt_id):
        return self.get_json(f"/history/{prompt_id}").get(prompt_id)

    def wait(self, prompt_id, poll=POLL_SECONDS):
        """Block until the prompt has left the queue; raise with the server's
        own message if it failed. Ctrl+C presses the server's Interrupt."""
        misses = 0
        try:
            while True:
                try:
                    entry = self.history(prompt_id)
                    misses = 0
                except ServerError:
                    misses += 1              # the server is busy or restarting
                    if misses > 30:
                        raise
                    time.sleep(poll)
                    continue
                if entry is not None:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            try:
                self.interrupt()
            except ServerError:
                pass
            raise
        status = entry.get("status") or {}
        if status.get("status_str") == "error" or status.get("completed") is False:
            raise ServerError(_execution_error(prompt_id, status))
        return entry

    def image(self, filename, subfolder="", type="output"):
        from PIL import Image
        data = self._request("/view", query={"filename": filename, "subfolder": subfolder, "type": type})
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    def images(self, history, node_id):
        out = history.get("outputs", {}).get(node_id, {})
        return [self.image(i["filename"], i.get("subfolder", ""), i.get("type", "output"))
                for i in out.get("images", [])]

    def text(self, history, node_id):
        out = history.get("outputs", {}).get(node_id, {})
        text = out.get("text", [])
        return "\n".join(text) if isinstance(text, (list, tuple)) else str(text)

    def interrupt(self):
        self._request("/interrupt", data=b"{}", content_type="application/json")

    def free(self, unload_models=False, free_memory=False):
        self.post_json("/free", {"unload_models": bool(unload_models), "free_memory": bool(free_memory)})

    # -- checking --------------------------------------------------------- #

    def validate(self, graph):
        """Everything ``/object_info`` lets us check before queueing: the
        class exists, every input name exists, every required input is
        present, links point at a real node and output of the right type,
        combo values are in the list. Returns a list of messages."""
        errors = []
        for node_id, node in graph.items():
            cls = node.get("class_type")
            inputs = node.get("inputs", {})
            info = self.object_info(cls)
            if info is None:
                errors.append(f"{node_id}: the server has no node class {cls!r}")
                continue
            specs = _input_specs(info)
            required = (info.get("input") or {}).get("required", {})
            for name in required:
                if name not in inputs:
                    errors.append(f"{node_id} ({cls}): required input {name!r} is missing")
            for name, value in inputs.items():
                spec = specs.get(name)
                if spec is None:
                    errors.append(f"{node_id} ({cls}): no input named {name!r} "
                                  f"(it has {sorted(specs)})")
                    continue
                kind = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
                if is_link(value):
                    source, slot = value
                    if source not in graph:
                        errors.append(f"{node_id}.{name}: links to a node {source!r} that is not in the graph")
                        continue
                    source_info = self.object_info(graph[source].get("class_type"))
                    if source_info is None:
                        continue                      # reported on its own line
                    outputs = source_info.get("output") or []
                    if not isinstance(slot, int) or slot < 0 or slot >= len(outputs):
                        errors.append(f"{node_id}.{name}: {source} has no output {slot}")
                    elif isinstance(kind, str) and kind not in ("*", "COMBO") \
                            and outputs[slot] not in ("*", kind):
                        errors.append(f"{node_id}.{name}: expects {kind}, "
                                      f"{source}.{slot} gives {outputs[slot]}")
                    continue
                options = _combo_options(spec)
                if options is not None:
                    if (cls, name) not in FOLDER_COMBOS and value not in options:
                        shown = options[:6] + (["..."] if len(options) > 6 else [])
                        errors.append(f"{node_id} ({cls}).{name}: {value!r} is not one of {shown}")
                elif kind in ("INT", "FLOAT"):
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        errors.append(f"{node_id} ({cls}).{name}: {value!r} is not a number")
                elif kind == "STRING":
                    if not isinstance(value, str):
                        errors.append(f"{node_id} ({cls}).{name}: {value!r} is not a string")
                elif kind == "BOOLEAN":
                    if not isinstance(value, bool):
                        errors.append(f"{node_id} ({cls}).{name}: {value!r} is not a boolean")
                elif isinstance(kind, str) and kind.isupper():
                    errors.append(f"{node_id} ({cls}).{name}: a {kind} must come from another "
                                  f"node, not a literal {value!r}")
        return errors


# --------------------------------------------------------------------------- #
# the graphs
# --------------------------------------------------------------------------- #

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def link(node_id, slot=0):
    return [node_id, slot]


def is_link(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


def guidance(strength, method="auto", space="auto", start=0.15, end=0.9,
             max_gain=2.0, lowpass=0):
    """The inputs of Apply Latent Color Guidance, named as the node names them."""
    return {"strength": float(strength), "start_percent": float(start),
            "end_percent": float(end), "method": method, "space": space,
            "max_gain": float(max_gain), "lowpass": int(lowpass)}


def _graph(ckpt, positive, negative, seed, steps, cfg, sampler, scheduler, denoise,
           size, reference_image, guidance, grade, img2img):
    graph = {
        "ckpt": node("CheckpointLoaderSimple", ckpt_name=ckpt),
        "positive": node("CLIPTextEncode", text=positive, clip=link("ckpt", 1)),
        "negative": node("CLIPTextEncode", text=negative, clip=link("ckpt", 1)),
    }
    if reference_image is not None:
        graph["reference"] = node("LoadImage", image=reference_image)
        graph["reference_latent"] = node("VAEEncode", pixels=link("reference", 0), vae=link("ckpt", 2))

    if img2img:
        if reference_image is None:
            raise ValueError("img2img needs a reference image")
        latent = link("reference_latent", 0)
    else:
        graph["empty"] = node("EmptyLatentImage", width=int(size), height=int(size), batch_size=1)
        latent = link("empty", 0)

    model = link("ckpt", 0)
    if guidance is not None:
        if reference_image is None:
            raise ValueError("guidance needs a reference image to take its colours from")
        graph["guidance"] = node("ApplyLatentColorGuidance", model=model,
                                 reference_latent=link("reference_latent", 0), **guidance)
        model = link("guidance", 0)
    if grade is not None:
        graph["grade"] = node("LatentColorGrade", model=model, **grade)
        model = link("grade", 0)

    graph["sampler"] = node("KSampler", model=model, seed=int(seed), steps=int(steps),
                            cfg=float(cfg), sampler_name=sampler, scheduler=scheduler,
                            positive=link("positive", 0), negative=link("negative", 0),
                            latent_image=latent, denoise=float(denoise))
    graph["decode"] = node("VAEDecode", samples=link("sampler", 0), vae=link("ckpt", 2))
    graph[OUTPUT_NODE] = node("PreviewImage", images=link("decode", 0))
    return graph


def graph_txt2img(ckpt, positive, negative, seed, steps, cfg, sampler, scheduler,
                  denoise, size, reference_image=None, guidance=None, grade=None):
    """Empty latent in. A reference image is only needed when guidance is."""
    return _graph(ckpt, positive, negative, seed, steps, cfg, sampler, scheduler,
                  denoise, size, reference_image, guidance, grade, img2img=False)


def graph_img2img(ckpt, positive, negative, seed, steps, cfg, sampler, scheduler,
                  denoise, size, reference_image=None, guidance=None, grade=None):
    """The reference image, encoded, is the latent being denoised — and, if
    guidance is given, the colours it is held to."""
    return _graph(ckpt, positive, negative, seed, steps, cfg, sampler, scheduler,
                  denoise, size, reference_image, guidance, grade, img2img=True)


def graph_lut(reference_image, lut_name, strength):
    """Apply a .cube table to an uploaded image. No model involved."""
    return {
        "image": node("LoadImage", image=reference_image),
        "lut": node("ColorKitApplyLUT", image=link("image", 0), lut_name=lut_name,
                    strength=float(strength)),
        OUTPUT_NODE: node("PreviewImage", images=link("lut", 0)),
    }


def validate_structure(graph):
    """What can be checked with no server at all."""
    errors = []
    for node_id, item in graph.items():
        cls = item.get("class_type")
        if not isinstance(cls, str) or not cls:
            errors.append(f"{node_id}: class_type is missing")
        for name, value in item.get("inputs", {}).items():
            if is_link(value) and value[0] not in graph:
                errors.append(f"{node_id}.{name}: links to a node {value[0]!r} that is not in the graph")
            elif value is None:
                errors.append(f"{node_id}.{name}: is None")
    return errors


def describe(graph):
    lines = []
    for node_id, item in graph.items():
        parts = []
        for name, value in item.get("inputs", {}).items():
            if is_link(value):
                parts.append(f"{name}={value[0]}.{value[1]}")
            elif isinstance(value, str):
                shown = value if len(value) <= 48 else value[:45] + "..."
                parts.append(f"{name}={shown!r}")
            else:
                parts.append(f"{name}={value}")
        lines.append(f"      {node_id:<16} {item.get('class_type', '?'):<26} " + ", ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #

def preflight(server, checkpoints=None, classes=CORE_CLASSES + PACK_CLASSES):
    """Fail before anything is queued: the server answers, the node classes
    exist there, the checkpoints are ones it lists. Returns the resolved
    checkpoint names, keyed as given."""
    server.require()
    missing = [cls for cls in classes if server.object_info(cls) is None]
    if missing:
        hint = ("Is this pack installed in that ComfyUI's custom_nodes, and has it been "
                "restarted since?" if any(m in PACK_CLASSES for m in missing)
                else "Is that a current ComfyUI?")
        raise ServerError("the server does not know these node classes: "
                          + ", ".join(missing) + ". " + hint)
    resolved = {}
    for label, given in (checkpoints or {}).items():
        if given:
            resolved[label] = server.resolve_checkpoint(given)
            print(f"  {label}: {resolved[label]}")
    return resolved


def placeholder(size, index=0):
    """Stands in for a picture during a dry run. Noise rather than a flat
    colour, so the arithmetic downstream has something to measure."""
    return np.random.default_rng(index).integers(0, 256, (size, size, 3), dtype=np.uint8)


def png_bytes(u8):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(u8)).save(buffer, format="PNG")
    return buffer.getvalue()


class Session:
    """One generation at a time. Queue it, wait, read the picture back, rest.

    With ``dry_run`` set the same call validates the graph (against the
    server when it answers, structurally when it does not), prints it when
    ``verbose``, writes it as JSON when ``dump_dir`` is set, and hands back a
    placeholder. Nothing is queued.
    """

    def __init__(self, server, gap=2.0, dry_run=False, verbose=False, dump_dir=None):
        self.server = server
        self.gap = float(gap)
        self.dry_run = dry_run
        self.verbose = verbose
        self.dump_dir = dump_dir
        self.built = []          # (label, graph)
        self.errors = []
        self.queued = 0
        self.seconds = 0.0

    def generate(self, graph, label, size, rest=True):
        self.built.append((label, graph))
        if self.server.reachable:
            errors = self.server.validate(graph)
        else:
            errors = validate_structure(graph)
        self.errors.extend(f"{label}: {e}" for e in errors)

        if self.dry_run:
            if self.verbose:
                print(f"  [dry-run] {label}")
                print(describe(graph))
                for error in errors:
                    print(f"      !! {error}")
            if self.dump_dir:
                os.makedirs(self.dump_dir, exist_ok=True)
                name = f"{len(self.built):02d}-{re.sub(r'[^A-Za-z0-9._-]+', '_', label)}.json"
                with open(os.path.join(self.dump_dir, name), "w", encoding="utf-8") as fh:
                    json.dump(graph, fh, indent=2)
            return placeholder(size, len(self.built))

        if errors:
            raise ServerError(f"the graph for {label!r} is not valid:\n  " + "\n  ".join(errors))

        started = time.time()
        prompt_id = self.server.queue(graph)
        print(f"  queued  {label}  ({prompt_id[:8]})", flush=True)
        history = self.server.wait(prompt_id)
        images = self.server.images(history, OUTPUT_NODE)
        if not images:
            raise ServerError(f"{label}: the server returned no image "
                              f"(outputs: {list(history.get('outputs', {}))})")
        elapsed = time.time() - started
        self.seconds += elapsed
        self.queued += 1
        print(f"  done    {label}  {elapsed:.0f}s", flush=True)
        if rest and self.gap > 0:
            time.sleep(self.gap)
        return images[0]

    def finish_dry(self):
        """Report the dry pass; stop the run if anything was wrong."""
        where = (f"against http://{self.server.address}" if self.server.reachable
                 else "structurally only — no server answered")
        if self.errors:
            print(f"\n  {len(self.errors)} problem(s) in {len(self.built)} graphs ({where}):")
            for error in self.errors:
                print(f"    !! {error}")
            raise SystemExit(1)
        print(f"  {len(self.built)} graphs built and checked {where}; nothing queued")


class Generator:
    """The graphs for one checkpoint, with the sampler settings a script uses,
    run through a :class:`Session`."""

    def __init__(self, session, ckpt, size, positive, negative, steps, cfg,
                 sampler="dpmpp_2m", scheduler="karras", reference_image=None):
        self.session = session
        self.ckpt = ckpt
        self.size = int(size)
        self.positive = positive
        self.negative = negative
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.sampler = sampler
        self.scheduler = scheduler
        self.reference_image = reference_image

    def _settings(self, seed, steps, cfg, denoise, guidance, grade):
        return dict(ckpt=self.ckpt, positive=self.positive, negative=self.negative,
                    seed=seed, steps=self.steps if steps is None else steps,
                    cfg=self.cfg if cfg is None else cfg, sampler=self.sampler,
                    scheduler=self.scheduler, denoise=denoise, size=self.size,
                    reference_image=self.reference_image, guidance=guidance, grade=grade)

    def txt2img(self, label, seed, denoise=1.0, steps=None, cfg=None, guidance=None, grade=None):
        graph = graph_txt2img(**self._settings(seed, steps, cfg, denoise, guidance, grade))
        return self.session.generate(graph, label, self.size)

    def img2img(self, label, seed, denoise, steps=None, cfg=None, guidance=None, grade=None):
        graph = graph_img2img(**self._settings(seed, steps, cfg, denoise, guidance, grade))
        return self.session.generate(graph, label, self.size)


# --------------------------------------------------------------------------- #
# details
# --------------------------------------------------------------------------- #

def _input_specs(info):
    """required and optional inputs of a node, name -> spec. Hidden ones are
    filled in by the server and never sent."""
    specs = {}
    for section in ("optional", "required"):
        specs.update((info.get("input") or {}).get(section, {}) or {})
    return specs


def _combo_options(spec):
    """The choices of a list input, or None when it is not one. Both the
    classic ``[[...], {...}]`` form and ``["COMBO", {"options": [...]}]``."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], (list, tuple)):
        return list(spec[0])
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return list(spec[1].get("options", []))
    return None


def _multipart(fields, files):
    boundary = "----ColorKit" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode("utf-8")
    for name, (filename, data, content_type) in files.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").encode("utf-8")
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _pretty_error(body):
    """The server's 400 body, readable: the top-level error and each node's."""
    try:
        data = json.loads(body)
    except ValueError:
        return body[:600]
    if not isinstance(data, dict):
        return body[:600]
    lines = []
    error = data.get("error")
    if isinstance(error, dict):
        lines.append(f"{error.get('message', '')}: {error.get('details', '')}".strip(": "))
    elif error:
        lines.append(str(error))
    for node_id, detail in (data.get("node_errors") or {}).items():
        cls = detail.get("class_type", "?") if isinstance(detail, dict) else "?"
        for item in (detail.get("errors", []) if isinstance(detail, dict) else []):
            lines.append(f"  node {node_id} ({cls}): {item.get('message', '')} "
                         f"{item.get('details', '')}".rstrip())
    return "\n".join(lines) if lines else body[:600]


def _execution_error(prompt_id, status):
    for event, data in status.get("messages", []):
        if event == "execution_error":
            return (f"prompt {prompt_id[:8]} failed in {data.get('node_type')} "
                    f"(node {data.get('node_id')}): {data.get('exception_message', '')}")
        if event == "execution_interrupted":
            return (f"prompt {prompt_id[:8]} was interrupted at {data.get('node_type')} "
                    f"(node {data.get('node_id')})")
    return f"prompt {prompt_id[:8]} did not complete: {json.dumps(status)[:600]}"
