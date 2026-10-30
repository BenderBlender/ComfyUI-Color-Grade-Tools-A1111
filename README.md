# A1111 Color Correction for ComfyUI

Keep a render's colours where you put them.

Any img2img, refine or upscale pass drifts in colour. The higher the denoise,
the further it goes: skies warm up, skin cools down, a grade you spent an hour
on comes back looking like someone else's. The Automatic1111 webui has had a
one-checkbox fix for this since 2022. This pack brings that fix to ComfyUI —
and then does the thing A1111 cannot: holds the colour *while the image is
being made* instead of repairing it afterwards.

**No models to download. No dependencies beyond what ComfyUI already installs.**

---

## What it does

### The blend decides whether the cast is removed at all

![Blend modes](docs/demos/01-blend-modes.png)

`A1111 Color Correction` is the faithful port of
`modules/processing.py: apply_color_correction`: histogram-match the result to
the reference in LAB, then blend so the generated image **keeps its own
brightness and takes only the reference's colour**.

That last clause is the whole operation, and it is easy to get backwards. The
third and fourth panels above are the *same* histogram match, blended the two
possible ways — 2.6 against 3.4 in distance from the reference. One removes
the cast, the other keeps most of it.

This matters here because the first version of these nodes had it backwards by
default. Over random image pairs the two blends differ by an average of
**78.6 / 255**. The old behaviour is still available as `lab_luminosity`,
honestly labelled.

### The whole strength range is useful

![Strength](docs/demos/02-image-strength.png)

### The colour is held during sampling, not patched afterwards

`Apply Latent Color Guidance` patches the `MODEL`. At every step the model
predicts what it thinks the finished image looks like; the patch nudges that
prediction towards your reference's palette and hands it back. The model then
keeps working *in* the corrected palette rather than being dragged into it at
the end — so the image is composed in the right colours, including the parts a
post-hoc filter can never reach, like what the model decides to put in the
shadows.

Because it patches the model instead of replacing the sampler, **every sampler
works**: KSampler, KSamplerAdvanced, SamplerCustom, detailers, anything
downstream.

| SD 1.5, img2img at denoise 0.75 | Distance from reference | Detail energy |
|---|---|---|
| no guidance | 6.09 | 6.81 |
| guidance, strength 0.6 | **4.76** (−21.8%) | 7.60 (+11.6%) |

Measured on the latent directly, without the VAE round trip in the way, the
correction removes **100%** of a deliberate cast (projected colour error
0.1767 → 0.0000).

### It works on every model family, with no per-model settings

SD1.5 and SDXL have 4-channel latents whose channels behave roughly like
colour axes. SD3 and Flux have 16, Wan 2.2 has 48, Flux.2 and LTX-Video have
128 — and there, only a few directions carry colour while the rest carry
structure and texture. Matching all of them per channel changes *how much
detail the image has*, which is not what anyone asks a colour node for.

So the correction is not done in raw channels. ComfyUI ships a latent-to-RGB
matrix for 42 of its 45 latent formats — the one it uses to draw sampling
previews. Projecting through it gives one algorithm for every model:

```
rgb   = Pᵀ·l + b                   project the latent into approximate RGB
rgb'  = f(rgb)                     do the colour transfer there, in 3 channels
Δl    = P(PᵀP)⁻¹·(rgb' − rgb)      minimum-norm step back into the latent
```

The pseudo-inverse is the point. Three equations, C unknowns: the minimum-norm
solution puts the entire correction inside `span(P)` — the three directions
that carry colour — and leaves the other C−3, which hold structure and
texture, untouched by construction.

Verified exactly (`Pᵀ·Δl == Δrgb` to 1e-6) for 4-, 16- and 48-channel latents,
and the 128↔32 channel repacking Flux.2 needs round-trips bit-for-bit.

### Three transfer methods, for three different problems

- **`mean_only`** shifts each channel's average. It cannot change contrast or
  detail, so it is the safe default on many-channel latents.
- **`reinhard`** matches average *and* spread — the classic transfer. Use it
  when the render has lost contrast as well as colour.
- **`mkl`** is optimal linear transport (Monge–Kantorovich). It accounts for
  correlations *between* channels, so a cast lying diagonally across them is
  removed rather than approximated axis by axis.

### The thing people remember as "built into sampling"

A1111 has **no colour correction inside its sampler**. Reading
`sd_samplers_cfg_denoiser.py`, `sd_samplers_kdiffusion.py` and
`sd_samplers_common.py` turns up no correction on any step; the only per-step
operation touching the latent is the inpaint mask blend.

What people remember is `scripts/loopback.py`, and **A1111 Loopback**
reproduces it: a chain of img2img passes where each output becomes the next
input, and every pass is corrected against one reference measured once from
the starting image. Because each pass is a full img2img at 0.75 → 0.5 denoise,
the *content* is reworked every time — which is exactly why it felt like more
than a colour filter — while the anchor never moves, so the palette cannot
drift however long the chain runs.

Testing it turned up a property of A1111's algorithm worth knowing about: its
blend takes lightness from the generated image, so a chain holds **colour**
perfectly flat and lets **exposure** drift uncorrected. Over ten passes on a
synthetic cast, chroma error stayed at 0.16 while lightness went to 28. The
node adds an optional `hold_exposure` for that, clearly marked as going beyond
A1111 — it matches lightness by average and spread rather than by its whole
histogram, so deliberate changes in lighting survive.

---

## The nodes

| Node | Takes | Gives | For |
|---|---|---|---|
| **Apply Latent Color Guidance** | `MODEL`, `LATENT` | `MODEL` | Holding a palette through sampling. The main one. |
| **A1111 Color Correction** | `IMAGE`, `IMAGE` | `IMAGE` | The faithful A1111 post-pass on a decoded image. |
| **A1111 Loopback** | `MODEL`, `VAE`, `IMAGE` | `IMAGE` | A chain of img2img passes anchored to one reference. |
| **Latent Color Match** | `LATENT`, `LATENT` | `LATENT` | One-shot match, before or after sampling. |
| **KSampler (A1111 Latent Color Match)** | — | — | Deprecated. Kept so old workflows still open. |

All of them take an optional `MASK`, which restricts *where the colours are
measured* — the usual want for inpainting, where the surrounding plate would
otherwise drag the statistics towards doing nothing.

---

## Install

**ComfyUI Manager** — search for "A1111 Color Correction".

**Manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_GITHUB_USERNAME/ComfyUI-A1111-ColorCorrection
```

Nothing to install afterwards. OpenCV is used automatically when present,
because A1111 runs on it and that makes the LAB conversion bit-identical;
without it the numpy path lands within 3/255 of the same place.

---

## Quick start

Three workflows are in [`examples/`](examples) — drag any into ComfyUI:

1. **`01_img2img_color_correction.json`** — the post-pass, with a before/after
   preview so you can see what it did.
2. **`02_latent_color_guidance.json`** — the model patch. Bypass it with
   Ctrl+B to compare.
3. **`03_guidance_plus_correction.json`** — both stages, the usual production
   setup for an upscale or refine pass.
4. **`04_a1111_loopback.json`** — the Loopback chain, with every pass
   previewed so you can watch the content change while the palette does not.

| Setting | Start at | Why |
|---|---|---|
| `strength` | 0.5–0.7 | Above that the model fights the correction instead of following it. |
| `start_percent` | 0.15 | Before that the prediction is still featureless and its spread means nothing. |
| `end_percent` | 0.9 | Leaving the last steps alone lets detail settle undisturbed. |
| `method` | `auto` | `reinhard` for 4-channel latents, `mean_only` where channels are not colour axes. |
| `space` | `auto` | The RGB projection — what makes this model-agnostic. |

---

## Measured results

"Distance" is the mean absolute difference between a result's LAB channel
statistics and the reference's, in 0–255 levels: lower is closer to the
colours you asked for.

**Image node, deliberate red/blue cast** (SD 1.5, 512):

| | Distance | |
|---|---|---|
| the cast | 12.35 | |
| corrected | **1.63** | 86.8% removed |

Brightness moves by 0.00097 — the A1111 blend keeps the original's luma, as it
should.

**Blend and strength** (SDXL, 1024, the figures above):

| strength | 0 | 0.35 | 0.7 | 1.0 |
|---|---|---|---|---|
| distance | 5.6 | 4.6 | 3.5 | **2.6** |

The inverted blend at full strength reaches only 3.4.

**Loopback chain** (SD 1.5, 512, three passes at denoise 0.65 → 0.45, drift
from the starting image):

| | Lightness | Chroma |
|---|---|---|
| no correction | 5.93 | 4.77 |
| A1111's correction | 5.51 | **0.13** |
| plus `hold_exposure` | **0.04** | 0.17 |

The middle row is the shape of A1111's algorithm, not a shortcoming of the
port: colour is pinned, exposure is left alone. Over a ten-pass synthetic
chain chroma error stays flat — 0.08 after two passes, 0.16 after ten — while
lightness climbs with every pass unless `hold_exposure` is raised.

### Every sampler, not most of them

The original node modified the sampler's working tensor in place from a
callback — which several samplers simply discard, so it silently did nothing
for them. The patch hooks the noise prediction instead. Calls counted over a
six-step run:

| Sampler | Calls | Sampler | Calls |
|---|---|---|---|
| euler | 6 | ddim | 6 |
| euler_ancestral | 6 | uni_pc | 6 |
| heun | 11 | lcm | 6 |
| dpm_2 | 11 | dpm_adaptive | 57 |
| dpmpp_2m | 6 | res_multistep | 6 |
| dpmpp_sde | 11 | | |

Samplers that evaluate the model twice per step correct twice per step, so
`strength` is per model call rather than per step. `dpm_adaptive` chooses its
own step count.

### Tests

68 unit tests, needing only numpy and torch:

```bash
python tests/test_color.py          # image maths, incl. A1111 parity   (14)
python tests/test_latent_core.py    # statistics, transfers, projection (30)
python tests/test_loopback.py       # the chain, with sampler and VAE stubbed (24)
python tests/test_examples.py       # the example workflows stay loadable
```

The integration suite drives real checkpoints inside ComfyUI and reports VRAM
and RAM at every boundary, so a run says what it costs and proves the models
were unloaded again:

```bash
python tests/test_integration_comfy.py --comfy /path/to/ComfyUI \
    --sd15 /path/to/checkpoint.safetensors --image reference.png
```

The demonstration figures are regenerated by `tools/make_demos.py`. It loads
large checkpoints and samples repeatedly — expect it to occupy the GPU for
several minutes.

---

## What changed from the original nodes

These began as two private nodes. Both had bugs worth naming, since you may
have workflows built on them.

**The blend was inverted** — see above. 78.6/255 of difference, by default.

**The reference latent was in the wrong scale.** The sampler works on latents
multiplied by the format's `scale_factor`, but a `LATENT` socket carries the
unscaled value — measured at 0.18215 for SD1.5, so a factor of 5.5 out, and
7.7 for SDXL. That is why the old node's strength defaulted to `0.01` and fell
apart at any sane value.

**Statistics came from the noisy latent.** Since `std(x)² = std(x₀)² + σ²`,
early in sampling that measurement is mostly noise. The correction now works
on the model's clean prediction, where the colour actually is.

Also fixed: `noise_mask` was ignored, so the node stopped being an inpaint
sampler; latent previews were lost; an even blur radius silently did nothing;
`requirements.txt` was cp1251-encoded, which makes pip fail on Linux and would
have blocked ComfyUI-Manager installs entirely.

### Migrating

`KSampler (A1111 Latent Color Match)` still loads and still samples. It is
marked deprecated and hidden from the menu, with the same inputs in the same
order, so saved graphs open unchanged. **Its strength now means something
different**: it corrects a correctly-scaled reference, so start around 0.3–0.5
rather than 0.01. The step ratios are now read as a share of the model's sigma
schedule rather than of the step count, which shifts the window slightly on
non-linear schedulers such as karras.

---

## Resolutions and model support

These nodes impose no resolution rule of their own, and the reference does not
have to match the size of what you are sampling — only its colour statistics
are read, so a 512×512 reference can steer a 1536×1536 upscale. It does have
to come from the same VAE. Per-family constraints are in
[`docs/model-support.md`](docs/model-support.md).

---

## Credits

The image pipeline reproduces `apply_color_correction` from
[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
including a numpy port of the `blendmodes` luminosity blend it relies on.

MIT licensed — see [LICENSE](LICENSE).
