// A grading panel on the node: three large wheels, laid out the way a
// colourist expects them.
//
// The numbers behind the wheels are ordinary FLOAT widgets (lift_r, lift_g,
// lift_b, lift_master and so on), so the nodes stay usable with this file
// absent or blocked — the panel is a way to reach those numbers, not a
// replacement for them.
//
// Dragging is relative, as it is in a grading suite: the handle moves with
// the pointer rather than jumping under it, which is the only way to make a
// small adjustment to a value that is already set. Hold Shift for a fifth of
// the sensitivity. Near the centre the handle snaps to neutral, so "no
// change" is a value you can actually return to.
//
// The geometry matches a1111_cc/grading.py exactly — red to the right, green
// at 120 degrees counter-clockwise, blue at 240, with the mean taken out so
// the disc only changes hue and the bar underneath handles brightness.
// tests/test_grading.py checks the two against each other.

import { app } from "../../scripts/app.js";

const NODES = ["LatentColorGrade", "ImageColorGrade"];
const WHEELS = [
  { name: "lift", label: "LIFT", hint: "shadows", centre: 0.0 },
  { name: "gamma", label: "GAMMA", hint: "midtones", centre: 1.0 },
  { name: "gain", label: "GAIN", hint: "highlights", centre: 1.0 },
];
const CHANNELS = ["r", "g", "b"];
const SPAN = 0.5;
const TAU = Math.PI * 2;

const DISC_MIN = 150;
const DISC_MAX = 260;
const GAP = 18;
const SIDE = 14;
const TITLE_H = 24;
const BAR_H = 22;
const READOUT_H = 20;
const DEAD_ZONE = 0.045;
const FINE = 0.2;

let discCache = null;
let discCacheSize = 0;

function widgetOf(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function readWheel(node, wheel) {
  return {
    rgb: CHANNELS.map((c) => widgetOf(node, `${wheel.name}_${c}`)?.value ?? 0),
    master: widgetOf(node, `${wheel.name}_master`)?.value ?? wheel.centre,
  };
}

function handleFromRgb(rgb) {
  let x = 0;
  let y = 0;
  for (let k = 0; k < 3; k += 1) {
    const phi = (k * TAU) / 3;
    x += rgb[k] * Math.cos(phi);
    y += rgb[k] * Math.sin(phi);
  }
  return { x: ((2 / 3) * x) / SPAN, y: ((2 / 3) * y) / SPAN };
}

function rgbFromHandle(x, y) {
  const angle = Math.atan2(y, x);
  const radius = Math.min(1, Math.hypot(x, y));
  if (radius < DEAD_ZONE) return [0, 0, 0];
  const raw = [0, 1, 2].map((k) => radius * Math.cos(angle - (k * TAU) / 3));
  const mean = (raw[0] + raw[1] + raw[2]) / 3;
  return raw.map((v) => (v - mean) * SPAN);
}

function setWheel(node, wheel, rgb, master) {
  CHANNELS.forEach((c, i) => {
    const w = widgetOf(node, `${wheel.name}_${c}`);
    if (w) {
      w.value = Math.max(-SPAN, Math.min(SPAN, rgb[i]));
      w.callback?.(w.value);
    }
  });
  if (master !== undefined) {
    const w = widgetOf(node, `${wheel.name}_master`);
    if (w) {
      w.value = Math.max(wheel.centre - SPAN, Math.min(wheel.centre + SPAN, master));
      w.callback?.(w.value);
    }
  }
  node.setDirtyCanvas(true, true);
}

function applyPreset(node, values) {
  for (const [name, value] of Object.entries(values)) {
    const w = widgetOf(node, name);
    if (w && typeof value === "number") {
      w.value = value;
      w.callback?.(w.value);
    }
  }
  node.setDirtyCanvas(true, true);
  node.graph?.afterChange?.();
}

function currentValues(node) {
  const out = {};
  for (const wheel of WHEELS) {
    for (const suffix of [...CHANNELS, "master"]) {
      const name = `${wheel.name}_${suffix}`;
      const w = widgetOf(node, name);
      if (w) out[name] = w.value;
    }
  }
  for (const name of ["saturation", "contrast"]) {
    const w = widgetOf(node, name);
    if (w) out[name] = w.value;
  }
  return out;
}

// The disc never changes, so it is painted once into an offscreen canvas and
// blitted afterwards — three of them redrawn per animation frame otherwise.
function discImage(size) {
  if (discCache && discCacheSize === size) return discCache;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const r = size / 2;
  const image = ctx.createImageData(size, size);
  const data = image.data;

  for (let py = 0; py < size; py += 1) {
    for (let px = 0; px < size; px += 1) {
      const dx = (px - r + 0.5) / r;
      const dy = (py - r + 0.5) / r;
      const radius = Math.hypot(dx, dy);
      const index = (py * size + px) * 4;
      if (radius > 1) {
        data[index + 3] = 0;
        continue;
      }
      const angle = Math.atan2(dy, dx);
      for (let k = 0; k < 3; k += 1) {
        const v = Math.cos(angle - (k * TAU) / 3);
        // Saturating towards the rim, neutral grey in the middle, so the
        // centre reads as "no change".
        data[index + k] = Math.round(132 + 112 * v * radius);
      }
      const edge = radius > 0.985 ? (1 - radius) / 0.015 : 1;
      data[index + 3] = Math.round(255 * Math.max(0, Math.min(1, edge)));
    }
  }
  ctx.putImageData(image, 0, 0);
  discCache = canvas;
  discCacheSize = size;
  return canvas;
}

function layout(width) {
  const usable = Math.max(240, width - SIDE * 2 - GAP * 2);
  const disc = Math.max(DISC_MIN, Math.min(DISC_MAX, Math.floor(usable / 3)));
  return { disc, height: TITLE_H + disc + BAR_H + READOUT_H + 10 };
}

function wheelRects(width, y) {
  const { disc } = layout(width);
  const total = disc * 3 + GAP * 2;
  const left = Math.max(SIDE, (width - total) / 2);
  return WHEELS.map((wheel, i) => {
    const x = left + i * (disc + GAP);
    return {
      wheel,
      disc,
      cx: x + disc / 2,
      cy: y + TITLE_H + disc / 2,
      radius: disc / 2 - 2,
      bar: { x: x + 10, y: y + TITLE_H + disc + 6, w: disc - 20, h: BAR_H - 12 },
      readoutY: y + TITLE_H + disc + BAR_H + 8,
    };
  });
}

function makeWidget() {
  const drag = { rect: null, kind: null, from: null, start: null, changed: false };

  const widget = {
    name: "colour_wheels",
    type: "colorkit_wheels",
    options: { serialize: false },
    value: null,

    computeSize(width) {
      return [width, layout(width).height];
    },

    draw(ctx, node, width, y) {
      const rects = wheelRects(width, y);
      ctx.save();
      ctx.textAlign = "center";

      for (const r of rects) {
        const { rgb, master } = readWheel(node, r.wheel);
        const handle = handleFromRgb(rgb);
        const touched = Math.hypot(handle.x, handle.y) > 1e-6 ||
          Math.abs(master - r.wheel.centre) > 1e-6;

        ctx.fillStyle = touched ? "#e8e8ef" : "#9a9aa4";
        ctx.font = "600 13px sans-serif";
        ctx.fillText(`${r.wheel.label}   ${r.wheel.hint}`, r.cx, y + 16);

        const size = Math.round(r.radius * 2);
        ctx.drawImage(discImage(size), r.cx - r.radius, r.cy - r.radius, size, size);

        ctx.beginPath();
        ctx.arc(r.cx, r.cy, r.radius, 0, TAU);
        ctx.strokeStyle = touched ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.5)";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // crosshair at neutral
        ctx.beginPath();
        ctx.moveTo(r.cx - 6, r.cy);
        ctx.lineTo(r.cx + 6, r.cy);
        ctx.moveTo(r.cx, r.cy - 6);
        ctx.lineTo(r.cx, r.cy + 6);
        ctx.strokeStyle = "rgba(0,0,0,0.35)";
        ctx.lineWidth = 1;
        ctx.stroke();

        const hx = r.cx + handle.x * r.radius;
        const hy = r.cy + handle.y * r.radius;
        ctx.beginPath();
        ctx.moveTo(r.cx, r.cy);
        ctx.lineTo(hx, hy);
        ctx.strokeStyle = "rgba(255,255,255,0.45)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(hx, hy, 8, 0, TAU);
        ctx.fillStyle = "rgba(250,250,255,0.95)";
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.8)";
        ctx.lineWidth = 2;
        ctx.stroke();

        // master bar
        const bar = r.bar;
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.beginPath();
        ctx.roundRect(bar.x, bar.y, bar.w, bar.h, bar.h / 2);
        ctx.fill();
        const centreX = bar.x + bar.w / 2;
        ctx.beginPath();
        ctx.moveTo(centreX, bar.y);
        ctx.lineTo(centreX, bar.y + bar.h);
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 1;
        ctx.stroke();

        const t = (master - (r.wheel.centre - SPAN)) / (SPAN * 2);
        const knob = bar.x + Math.max(0, Math.min(1, t)) * bar.w;
        ctx.beginPath();
        ctx.arc(knob, bar.y + bar.h / 2, bar.h / 2 + 2, 0, TAU);
        ctx.fillStyle = "#e2e2e8";
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.65)";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = "#8f8f99";
        ctx.font = "10px monospace";
        ctx.fillText(
          `${rgb.map((v) => (v >= 0 ? " " : "") + v.toFixed(3)).join(" ")}   M ${master.toFixed(3)}`,
          r.cx, r.readoutY,
        );
      }
      ctx.restore();
    },

    mouse(event, pos, node) {
      const rects = wheelRects(node.size[0], widget.last_y ?? 0);
      const [mx, my] = pos;

      if (event.type === "pointerdown") {
        for (const r of rects) {
          if (Math.hypot(mx - r.cx, my - r.cy) <= r.radius + 6) {
            if (event.altKey) {
              setWheel(node, r.wheel, [0, 0, 0], r.wheel.centre);
              node.graph?.afterChange?.();
              return true;
            }
            const { rgb } = readWheel(node, r.wheel);
            drag.rect = r;
            drag.kind = "disc";
            drag.from = handleFromRgb(rgb);
            drag.start = [mx, my];
            drag.changed = false;
            return true;
          }
          const bar = r.bar;
          if (mx >= bar.x - 6 && mx <= bar.x + bar.w + 6 &&
              my >= bar.y - 8 && my <= bar.y + bar.h + 8) {
            if (event.altKey) {
              const { rgb } = readWheel(node, r.wheel);
              setWheel(node, r.wheel, rgb, r.wheel.centre);
              node.graph?.afterChange?.();
              return true;
            }
            drag.rect = r;
            drag.kind = "bar";
            drag.from = readWheel(node, r.wheel).master;
            drag.start = [mx, my];
            drag.changed = false;
            return true;
          }
        }
        return false;
      }

      if (event.type === "pointermove" && drag.rect) {
        const r = drag.rect;
        const scale = event.shiftKey ? FINE : 1.0;
        drag.changed = true;
        if (drag.kind === "disc") {
          const dx = ((mx - drag.start[0]) / r.radius) * scale;
          const dy = ((my - drag.start[1]) / r.radius) * scale;
          let x = drag.from.x + dx;
          let y = drag.from.y + dy;
          const radius = Math.hypot(x, y);
          if (radius > 1) {
            x /= radius;
            y /= radius;
          }
          setWheel(node, r.wheel, rgbFromHandle(x, y));
        } else {
          const dx = ((mx - drag.start[0]) / r.bar.w) * SPAN * 2 * scale;
          const { rgb } = readWheel(node, r.wheel);
          setWheel(node, r.wheel, rgb, drag.from + dx);
        }
        return true;
      }

      if (event.type === "pointerup" && drag.rect) {
        if (drag.changed) node.graph?.afterChange?.();   // one undo step per drag
        drag.rect = null;
        drag.kind = null;
        return true;
      }
      return false;
    },
  };

  return widget;
}

// --------------------------------------------------------------------------- //
// presets
// --------------------------------------------------------------------------- //

async function fetchPresets() {
  try {
    const response = await fetch("/colorkit/presets");
    if (!response.ok) throw new Error(response.statusText);
    return await response.json();
  } catch (err) {
    console.warn("[ColorKit] could not read presets:", err);
    return { builtin: {}, user: {} };
  }
}

async function savePreset(name, values) {
  const response = await fetch("/colorkit/presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, values }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error ?? response.statusText);
  }
}

async function deletePreset(name) {
  await fetch(`/colorkit/presets?name=${encodeURIComponent(name)}`, { method: "DELETE" });
}

function presetMenu(node, entries, onPick) {
  const names = Object.keys(entries);
  if (!names.length) return [{ content: "(none yet)", disabled: true }];
  return names.map((name) => ({
    content: name,
    callback: () => onPick(name, entries[name]),
  }));
}

app.registerExtension({
  name: "colorkit.wheels",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      this.colorkitHidden = [];
      for (const wheel of WHEELS) {
        for (const suffix of [...CHANNELS, "master"]) {
          const w = widgetOf(this, `${wheel.name}_${suffix}`);
          if (w) this.colorkitHidden.push(w);
        }
      }
      this.colorkitShowNumbers = false;
      hideNumeric(this);

      const wheels = makeWidget();
      this.addCustomWidget(wheels);
      const index = this.widgets.indexOf(wheels);
      if (index > 0) {
        this.widgets.splice(index, 1);
        this.widgets.unshift(wheels);
      }

      this.size[0] = Math.max(this.size[0], 640);
      this.size[1] = this.computeSize()[1];
      return result;
    };

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      getExtraMenuOptions?.apply(this, arguments);
      const node = this;

      options.push(
        {
          content: "Presets",
          has_submenu: true,
          callback: async (value, opts, event, parentMenu) => {
            const { builtin, user } = await fetchPresets();
            const entries = { ...builtin, ...user };
            new LiteGraph.ContextMenu(
              presetMenu(node, entries, (name, values) => applyPreset(node, values)),
              { event, parentMenu, title: "Apply a preset" },
            );
          },
        },
        {
          content: "Save these settings as a preset…",
          callback: () => {
            canvas.prompt("Preset name", "", async (name) => {
              if (!name) return;
              try {
                await savePreset(name, currentValues(node));
              } catch (err) {
                alert(`Could not save the preset: ${err.message}`);
              }
            }, {});
          },
        },
        {
          content: "Delete a preset",
          has_submenu: true,
          callback: async (value, opts, event, parentMenu) => {
            const { user } = await fetchPresets();
            new LiteGraph.ContextMenu(
              presetMenu(node, user, async (name) => { await deletePreset(name); }),
              { event, parentMenu, title: "Delete which?" },
            );
          },
        },
        {
          content: node.colorkitShowNumbers ? "Hide the numbers" : "Show the numbers",
          callback: () => {
            node.colorkitShowNumbers = !node.colorkitShowNumbers;
            if (node.colorkitShowNumbers) showNumeric(node);
            else hideNumeric(node);
            node.size[1] = node.computeSize()[1];
            node.setDirtyCanvas(true, true);
          },
        },
        {
          content: "Reset all three wheels",
          callback: () => {
            for (const wheel of WHEELS) setWheel(node, wheel, [0, 0, 0], wheel.centre);
            node.graph?.afterChange?.();
          },
        },
      );
    };
  },
});

function hideNumeric(node) {
  for (const w of node.colorkitHidden ?? []) {
    w.colorkitType = w.colorkitType ?? w.type;
    w.type = "colorkit_hidden";
    w.computeSize = () => [0, -4];
  }
}

function showNumeric(node) {
  for (const w of node.colorkitHidden ?? []) {
    if (w.colorkitType) w.type = w.colorkitType;
    delete w.computeSize;
  }
}
