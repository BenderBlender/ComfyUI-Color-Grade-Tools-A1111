// Three colour wheels for the grading nodes, drawn on the node itself.
//
// The numbers behind them are ordinary FLOAT widgets (lift_r, lift_g, lift_b,
// lift_master and so on), so the nodes stay usable with this file absent or
// blocked — the wheels are a way to reach those numbers, not a replacement
// for them. Dragging a handle writes into the three channel widgets; the ring
// under each disc is its master.
//
// The mapping matches a1111_cc/grading.py exactly: red to the right, green at
// 120 degrees counter-clockwise, blue at 240, and the mean taken out so the
// disc only ever changes hue while the ring handles brightness.

import { app } from "../../scripts/app.js";

const NODES = ["LatentColorGrade", "ImageColorGrade"];
const WHEELS = [
  { name: "lift", label: "Lift", hint: "shadows", centre: 0.0 },
  { name: "gamma", label: "Gamma", hint: "midtones", centre: 1.0 },
  { name: "gain", label: "Gain", hint: "highlights", centre: 1.0 },
];
const CHANNELS = ["r", "g", "b"];
const SPAN = 0.5;          // the range the channel widgets accept
const TAU = Math.PI * 2;

const DISC_MIN = 74;
const DISC_MAX = 128;
const GAP = 12;
const LABEL_H = 18;
const RING_H = 20;
const READOUT_H = 15;
const PAD = 8;

function widgetOf(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function readWheel(node, wheel) {
  const values = CHANNELS.map((c) => widgetOf(node, `${wheel.name}_${c}`)?.value ?? 0);
  const master = widgetOf(node, `${wheel.name}_master`)?.value ?? wheel.centre;
  return { rgb: values, master };
}

// rgb offsets -> where the handle sits, as a unit vector on the disc.
// The basis vectors sum to zero, so the mean taken out on the way in does not
// disturb this: the inverse is exact.
function handleFromRgb(rgb) {
  let x = 0;
  let y = 0;
  for (let k = 0; k < 3; k += 1) {
    const phi = (k * TAU) / 3;
    x += rgb[k] * Math.cos(phi);
    y += rgb[k] * Math.sin(phi);
  }
  x = (2 / 3) * x / SPAN;
  y = (2 / 3) * y / SPAN;
  const radius = Math.hypot(x, y);
  return { x, y, radius: Math.min(radius, 1) };
}

// A point on the disc -> the three channel offsets.
function rgbFromHandle(x, y) {
  const angle = Math.atan2(y, x);
  const radius = Math.min(1, Math.hypot(x, y));
  const raw = [];
  for (let k = 0; k < 3; k += 1) {
    raw.push(radius * Math.cos(angle - (k * TAU) / 3));
  }
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

function drawDisc(ctx, cx, cy, radius) {
  // The hue circle, drawn as wedges rather than a conic gradient so it looks
  // the same in every browser the editor runs in.
  const steps = 72;
  for (let i = 0; i < steps; i += 1) {
    const a0 = (i / steps) * TAU;
    const a1 = ((i + 1) / steps) * TAU;
    const mid = (a0 + a1) / 2;
    const rgb = [0, 1, 2].map((k) => {
      const v = Math.cos(mid - (k * TAU) / 3);
      return Math.round(128 + 110 * v);
    });
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, a0, a1 + 0.02);
    ctx.closePath();
    ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    ctx.fill();
  }

  // Fade towards neutral in the middle, so "no change" reads as the centre.
  const fade = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  fade.addColorStop(0, "rgba(128,128,128,1)");
  fade.addColorStop(0.55, "rgba(128,128,128,0.25)");
  fade.addColorStop(1, "rgba(128,128,128,0)");
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, TAU);
  ctx.fillStyle = fade;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, TAU);
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx - 4, cy);
  ctx.lineTo(cx + 4, cy);
  ctx.moveTo(cx, cy - 4);
  ctx.lineTo(cx, cy + 4);
  ctx.strokeStyle = "rgba(0,0,0,0.4)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function layout(width) {
  const usable = width - PAD * 2 - GAP * 2;
  const disc = Math.max(DISC_MIN, Math.min(DISC_MAX, usable / 3));
  return { disc, height: LABEL_H + disc + RING_H + READOUT_H + PAD };
}

function wheelRects(node, width, y) {
  const { disc } = layout(width);
  const total = disc * 3 + GAP * 2;
  const left = (width - total) / 2;
  return WHEELS.map((wheel, i) => {
    const x = left + i * (disc + GAP);
    return {
      wheel,
      x,
      disc,
      cx: x + disc / 2,
      cy: y + LABEL_H + disc / 2,
      radius: disc / 2 - 3,
      ring: { x, y: y + LABEL_H + disc + 4, w: disc, h: RING_H - 8 },
    };
  });
}

function makeWidget(node) {
  const state = { dragging: null };

  const widget = {
    name: "colour_wheels",
    type: "a1111cc_wheels",
    options: { serialize: false },
    value: null,

    computeSize(width) {
      return [width, layout(width).height];
    },

    draw(ctx, drawnNode, width, y) {
      const rects = wheelRects(drawnNode, width, y);
      ctx.save();
      ctx.textAlign = "center";

      for (const r of rects) {
        const { rgb, master } = readWheel(drawnNode, r.wheel);
        const handle = handleFromRgb(rgb);

        ctx.fillStyle = "#cfcfd4";
        ctx.font = "600 11px sans-serif";
        ctx.fillText(`${r.wheel.label}  ·  ${r.wheel.hint}`, r.cx, y + 12);

        drawDisc(ctx, r.cx, r.cy, r.radius);

        const hx = r.cx + handle.x * r.radius;
        const hy = r.cy + handle.y * r.radius;
        ctx.beginPath();
        ctx.arc(hx, hy, 6, 0, TAU);
        ctx.fillStyle = "rgba(255,255,255,0.92)";
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // master ring
        const ring = r.ring;
        ctx.fillStyle = "rgba(0,0,0,0.35)";
        ctx.beginPath();
        ctx.roundRect(ring.x, ring.y, ring.w, ring.h, ring.h / 2);
        ctx.fill();
        const t = (master - (r.wheel.centre - SPAN)) / (SPAN * 2);
        const knob = ring.x + Math.max(0, Math.min(1, t)) * ring.w;
        ctx.beginPath();
        ctx.arc(knob, ring.y + ring.h / 2, ring.h / 2 + 1, 0, TAU);
        ctx.fillStyle = "#d8d8de";
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.6)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "#8d8d96";
        ctx.font = "9px monospace";
        const readout = `${rgb.map((v) => v.toFixed(2)).join(" ")}  ${master.toFixed(2)}`;
        ctx.fillText(readout, r.cx, ring.y + ring.h + 11);
      }
      ctx.restore();
    },

    mouse(event, pos, drawnNode) {
      const width = drawnNode.size[0];
      const rects = wheelRects(drawnNode, width, widget.last_y ?? 0);
      const [mx, my] = pos;

      if (event.type === "pointerdown") {
        for (const r of rects) {
          if (Math.hypot(mx - r.cx, my - r.cy) <= r.radius + 4) {
            state.dragging = { kind: "disc", rect: r };
            if (event.detail === 2 || event.shiftKey) {
              setWheel(drawnNode, r.wheel, [0, 0, 0], r.wheel.centre);
              state.dragging = null;
              return true;
            }
            applyDisc(drawnNode, r, mx, my);
            return true;
          }
          const ring = r.ring;
          if (mx >= ring.x && mx <= ring.x + ring.w &&
              my >= ring.y - 4 && my <= ring.y + ring.h + 4) {
            state.dragging = { kind: "ring", rect: r };
            applyRing(drawnNode, r, mx);
            return true;
          }
        }
        return false;
      }

      if (event.type === "pointermove" && state.dragging) {
        const r = state.dragging.rect;
        if (state.dragging.kind === "disc") applyDisc(drawnNode, r, mx, my);
        else applyRing(drawnNode, r, mx);
        return true;
      }

      if (event.type === "pointerup" && state.dragging) {
        state.dragging = null;
        return true;
      }
      return false;
    },
  };

  function applyDisc(target, rect, mx, my) {
    const x = (mx - rect.cx) / rect.radius;
    const y = (my - rect.cy) / rect.radius;
    setWheel(target, rect.wheel, rgbFromHandle(x, y));
  }

  function applyRing(target, rect, mx) {
    const ring = rect.ring;
    const t = Math.max(0, Math.min(1, (mx - ring.x) / ring.w));
    const value = rect.wheel.centre - SPAN + t * SPAN * 2;
    const { rgb } = readWheel(target, rect.wheel);
    setWheel(target, rect.wheel, rgb, value);
  }

  return widget;
}

app.registerExtension({
  name: "a1111cc.colorwheels",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      // The numeric widgets stay on the node and keep working; they are moved
      // out of sight so the wheels are what you see, and brought back by the
      // right-click entry below for anyone who wants to type exact values.
      this.a1111ccHidden = [];
      for (const wheel of WHEELS) {
        for (const suffix of [...CHANNELS, "master"]) {
          const w = widgetOf(this, `${wheel.name}_${suffix}`);
          if (w) this.a1111ccHidden.push(w);
        }
      }
      this.a1111ccShowNumbers = false;
      hideNumeric(this);

      const wheels = makeWidget(this);
      this.addCustomWidget(wheels);
      // Put the wheels above the remaining controls rather than at the end.
      const index = this.widgets.indexOf(wheels);
      if (index > 0) {
        this.widgets.splice(index, 1);
        this.widgets.unshift(wheels);
      }

      this.size[0] = Math.max(this.size[0], 330);
      this.size[1] = this.computeSize()[1];
      return result;
    };

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      getExtraMenuOptions?.apply(this, arguments);
      options.push(
        {
          content: this.a1111ccShowNumbers ? "Hide the numbers" : "Show the numbers",
          callback: () => {
            this.a1111ccShowNumbers = !this.a1111ccShowNumbers;
            if (this.a1111ccShowNumbers) showNumeric(this);
            else hideNumeric(this);
            this.size[1] = this.computeSize()[1];
            this.setDirtyCanvas(true, true);
          },
        },
        {
          content: "Reset all three wheels",
          callback: () => {
            for (const wheel of WHEELS) setWheel(this, wheel, [0, 0, 0], wheel.centre);
          },
        },
      );
    };
  },
});

function hideNumeric(node) {
  for (const w of node.a1111ccHidden ?? []) {
    w.a1111ccType = w.a1111ccType ?? w.type;
    w.type = "a1111cc_hidden";
    w.computeSize = () => [0, -4];
  }
}

function showNumeric(node) {
  for (const w of node.a1111ccHidden ?? []) {
    if (w.a1111ccType) w.type = w.a1111ccType;
    delete w.computeSize;
  }
}
