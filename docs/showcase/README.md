<p align="center"><img src="media/hero.jpg" alt="Color Grade Tools for ComfyUI" width="100%"></p>

# Color Grade Tools — the full tour

Colour wheels and film palettes that work **inside** the diffusion model while it draws, and the colour correction
Automatic1111 users have relied on since 2022, ported bit for bit. This page shows every node at work, explains how the
latent part works and why it matters, and gives the measured numbers behind every claim.

Everything below was rendered on 30–31 October 2026 through a running ComfyUI (SDXL ProtoVision XL unless a section
says otherwise) and measured with OpenCV LAB means. Click any picture for a larger one. The test pictures — a girl on a
porch (illustration), a mech robot (photo style), a pickup chased by a tree monster (photo style) and a witch's selfie
(painting) — are the author's own; the film frames are open-licensed, see [CREDITS](CREDITS.md).

**Contents** · [The idea](#the-idea-in-one-picture) · [Why the latent space](#why-grade-in-the-latent-space) ·
[The A1111 system](#the-a1111-system) · [Wheels inside sampling](#1-grade-wheels-inside-sampling) ·
[Any picture as a palette](#2-any-picture-becomes-a-palette) · [Palette plus your wheels](#3-palette-plus-your-own-wheels) ·
[Colour held through img2img](#4-colour-held-through-img2img) ·
[Loopback](#5-loopback-eight-img2img-passes) · [Wheels and presets](#6-grade-wheels-presets-and-luts) ·
[Other families](#7-other-model-families) · [IP-Adapter](#8-ip-adapter-and-colour-held) · [Masks](#9-masks) ·
[Three more pictures](#10-checked-on-three-more-pictures) · [Every node](#every-node-one-by-one) · [Try it](#try-it)

## The idea in one picture

<p align="center"><img src="media/palettes.webp" alt="One picture redrawn in seven film palettes" width="640"></p>

One girl, one seed, one prompt. The only thing that changes is the **colour source** — a frame from a film. The model
redraws the picture in that frame's palette: not a tint laid over the top, but light, shadow and colour decided while the
picture is being made. Template 08 does exactly this with two pictures of your own.

## Why grade in the latent space

### What a diffusion model actually works on

SDXL never paints pixels. A 1024 × 1024 picture lives as a 128 × 128 grid of four numbers per cell — the **latent** — and
only the VAE decoder, at the very end, turns it into pixels. Sampling is a loop: at every step the model looks at the
noisy latent and **predicts the finished picture**; the sampler moves part of the way towards that prediction, puts back
the right amount of noise and asks again. Twenty-odd predictions later the noise is gone.

Every colour tool you have used in ComfyUI works after that loop, on decoded pixels. It can tint what the model drew. It
cannot change what the model *decided* to draw — where the light falls, how deep the shadows go, what colour a dress
turned out to be.

### Where these nodes step in

<p align="center"><img src="media/latent_loop.svg" alt="The sampling loop with Color Grade inside it" width="100%"></p>

The latent nodes patch the `MODEL`. Each time the model returns its prediction of the finished picture, the patch
measures that prediction's colour, moves it part of the way towards a target and hands it back. The sampler then steps
from the corrected prediction, so **every following step is taken inside the target palette**. The picture is composed in
the look rather than recoloured into it.

The hook sits on the model's prediction (ComfyUI's `PREDICT_NOISE` wrapper), so it works under **every sampler and every
node that samples**: KSampler, KSamplerAdvanced, SamplerCustom, detailers, upscalers that resample. A window —
`start_percent` → `end_percent` of the model's schedule — chooses when it acts.

The target can come from three places:

- **Grade Wheels (latent)** — lift, gamma and gain wheels, saturation and contrast: a look you dial in.
- **Latent Color Guidance** — a reference picture's palette: the img2img input itself (so colour cannot drift), or any
  other picture.
- **Grade Wheels with an `anchor`** — the levels of a given latent, then your wheels on top.

### How colour is read out of a latent

A latent channel is not red, green or blue. For 42 of its 45 latent formats ComfyUI ships a small matrix that maps a
latent to approximate RGB — the one behind its live sampling previews. The nodes measure colour through it and write the
correction back with its pseudo-inverse, the minimum-norm step: the change stays inside the three directions that carry
colour, and the other channels — structure and texture on 16-channel SD3 and Flux latents, 48 on Wan, 128 on Flux.2 —
are not touched. That is what makes the same controls work on every model family.

For SDXL and SD3 the pack goes one step further. The preview matrix is a straight-line fit to a decoder that is not
straight; inverted, it turned warm highlights magenta at three to six times the asked strength. So the decoder itself
was measured — each latent channel pushed both ways, decoded, its colour response recorded, together with the decoder's
tone curve — and **Grade Wheels (latent) writes through those measured axes**.

The wheels read the levels of a lightly blurred prediction (5 latent pixels, so they grade areas, not texture), compute
the target once from the first prediction in the window, hold it, and close half of the remaining gap at each model
call — and they move only that blurred picture, leaving the fine part to the model. A latent pixel's projection is not
that pixel's colour: much of it is the latent's texture code, and a level map applied pixel by pixel rescaled it — neon
blotches under an anchor, fine detail ×0.72–1.37 without one. Moving only the blurred picture keeps the detail at
×0.87–1.04 on four test pictures. The grade neither piles up over twenty steps nor fades as the model keeps working;
`strength` scales it, though not exactly in proportion: 0.5 gives ×0.39 of the full shift on the girl,
×0.58–0.78 on three other pictures.

### Why it is worth having

- **The model draws in the look.** Highlights, shadows and the colours of things are decided inside the palette.
- **Any picture becomes a palette.** A film still, a painting, a photograph you like the light of.
- **Colour that does not drift.** img2img, refine and upscale passes keep the palette you started from.
- **Every sampler, every family.** One patch on the `MODEL`: SD 1.5, SDXL, SD3, Flux, video latents.
- **Nothing to download, nothing to train.** A few small matrix operations per model call; a 1024² SDXL img2img with the
  nodes in place took 4–6 s on an RTX 4090.

### What it does not promise

- **The exact hue and size are approximate**, because the decoder is not linear. Against the same wheels applied
  afterwards, the latent result lands 0.4–1.4 LAB units away on the girl (for shifts of 5–16), 2.2–3.6 on a photo of a
  robot and on a painted crowd — and 4.8–6.4 on a dark, green, snowy forest scene, where Golden hour came out greener
  rather than golden (section 10; the code before the current one did the same there). On SD 3.5 a street lands within
  8.3° at ×0.91.
- **Light passes start below the window.** At denoise ≈ 0.3 the run begins after `end_percent`, so the node does
  nothing and says so in the console. Use A1111 Color Correction there.
- **Grade Wheels + `anchor` is the gentler way to a frame's palette.** It takes the frame's levels and colourfulness and
  keeps the picture's detail (×0.81–1.23 on four pictures), but gets 0.43–0.96 of the way to the frame's colour and can
  oversaturate warm frames. The three-node chain of template 08 gets further, 0.62–0.99, at the price of darker,
  softer results on dark frames — and on the robot it rewrote the lettering on its chest.
- **When a number must be exact, finish on an image node** — they are exact by construction.

## The A1111 system

<p align="center"><img src="media/a1111_flow.svg" alt="A1111 colour correction: histogram match in LAB, then a luminosity blend" width="100%"></p>

### How it works

The Automatic1111 webui has a setting called *Apply color correction to img2img results to match original colors*.
Before sampling it stores the input picture in LAB colour space. After the img2img is decoded it matches the histogram of
each LAB channel — lightness **L**, green–red **a**, blue–yellow **b** — to the input's, converts back to RGB, and blends
the matched picture with the render in **luminosity** mode: brightness from the new render, colour from the match.

### Why it is interesting

- It is why long img2img chains in the webui — Loopback, upscale and refine passes — did not slide into a colour cast.
- It needs no model, no training and no tuning, and runs in a fraction of a second.
- It keeps the new render's light and puts back only the colour. Get the blend the wrong way round and it does nearly
  the opposite: the first version of these nodes did, by default — an average of 78.6 of 255 apart on random pairs.
- What people remember as A1111 correcting colour "inside sampling" is its **Loopback** script: a chain of img2img passes,
  each corrected against the first frame. That is here too.

### Ported, not imitated

**A1111 Color Correction** is the webui's `apply_color_correction` reproduced line for line and checked against the
webui's own code on the libraries it pins (numpy 1.26.2, OpenCV 4.8.1, scikit-image 0.21.0, blendmodes 2022):
**0 of 255** difference on four img2img pairs. Getting there meant matching details no screenshot reveals — the
truncation of the histogram lookup, Rec. 601 luma (0.299 / 0.587 / 0.114) in float64, the order of a summation.

## 1. Grade wheels inside sampling

The girl through SDXL img2img at denoise 0.5 (seed 0, dpmpp_2m / karras, 24 steps, cfg 7). Middle column: **Grade
Wheels (image)** on the finished render. Right column: the same wheels on **Grade Wheels (latent)**, given to the model
(strength 1.0, window 0.6 → 1.0).

<p align="center"><img src="media/inside_vs_after.jpg" alt="No grade, wheels after, wheels inside sampling" width="100%"></p>

| | Input | img2img, no grade | Wheels after | Wheels inside sampling |
|---|---|---|---|---|
| **Teal and orange** | <a href="frames/girl.jpg"><img src="frames/girl.jpg" alt="Input, Teal and orange" width="100%"></a> | <a href="frames/GW_girl_none.jpg"><img src="frames/GW_girl_none.jpg" alt="img2img, no grade, Teal and orange" width="100%"></a> | <a href="frames/GW_girl_teal_and_orange_after.jpg"><img src="frames/GW_girl_teal_and_orange_after.jpg" alt="Wheels after, Teal and orange" width="100%"></a> | <a href="frames/GW_girl_teal_and_orange_during_k1.0.jpg"><img src="frames/GW_girl_teal_and_orange_during_k1.0.jpg" alt="Wheels inside sampling, Teal and orange" width="100%"></a> |

The colour shift each makes from the ungraded render, latent against image: Teal and orange 11.4° apart
at ×0.88 the size, Golden hour 5.6° at ×1.09, Moonlight
1.1° at ×0.99. Fine detail stays close to the ungraded render's: ×0.92–1.01.

**Strength is a dial, not a switch.** Teal and orange at 0.5 gives ×0.39 of the shift at
1.0:

| | No grade | strength 0.5 | strength 1.0 |
|---|---|---|---|
| **Grade Wheels (latent)** | <a href="frames/GW_girl_none.jpg"><img src="frames/GW_girl_none.jpg" alt="No grade, Grade Wheels (latent)" width="100%"></a> | <a href="frames/GW_girl_teal_and_orange_during_k0.5.jpg"><img src="frames/GW_girl_teal_and_orange_during_k0.5.jpg" alt="strength 0.5, Grade Wheels (latent)" width="100%"></a> | <a href="frames/GW_girl_teal_and_orange_during_k1.0.jpg"><img src="frames/GW_girl_teal_and_orange_during_k1.0.jpg" alt="strength 1.0, Grade Wheels (latent)" width="100%"></a> |

**In the editor** — template 06, made img2img on the girl and run: the latent wheels patch the model (top left), the
same seed renders on the ungraded model below, and the image wheels grade that render afterwards (top right).

<a href="frames/ui_11_run_grade_while_sampling_girl.jpg"><img src="frames/ui_11_run_grade_while_sampling_girl.jpg" alt="Template 06 on the girl, run" width="100%"></a>

## 2. Any picture becomes a palette

<p align="center"><img src="media/palettes_grid.jpg" alt="The girl in seven film palettes" width="100%"></p>

Template 08 takes two pictures: yours, and a colour source. **Latent Color Match** moves the starting latent towards the
source, **Latent Color Guidance** (strength 0.5) holds that palette at every step, and **A1111 Color Correction**
(strength 0.8) finishes on the decoded picture — colour from the source, brightness from the render. Same settings as
section 1.

<a href="frames/ui_10_run_palette_from_a_picture.jpg"><img src="frames/ui_10_run_palette_from_a_picture.jpg" alt="Template 08 run: the girl in the palette of a frame from Spring" width="100%"></a>

| | Colour source | Latent Color Guidance alone | Match + Guidance + Correction (template 08) |
|---|---|---|---|
| **Tears of Steel — steel blue** | <a href="frames/frame_tos_teal.jpg"><img src="frames/frame_tos_teal.jpg" alt="Colour source, Tears of Steel — steel blue" width="100%"></a> | <a href="frames/GF_girl_ref-tos_teal_02.jpg"><img src="frames/GF_girl_ref-tos_teal_02.jpg" alt="Latent Color Guidance alone, Tears of Steel — steel blue" width="100%"></a> | <a href="frames/GF_girl_ref-tos_teal_03.jpg"><img src="frames/GF_girl_ref-tos_teal_03.jpg" alt="Match + Guidance + Correction (template 08), Tears of Steel — steel blue" width="100%"></a> |
| **Sintel — warm interior** | <a href="frames/frame_sintel_warm.jpg"><img src="frames/frame_sintel_warm.jpg" alt="Colour source, Sintel — warm interior" width="100%"></a> | <a href="frames/GF_girl_ref-sintel_warm_02.jpg"><img src="frames/GF_girl_ref-sintel_warm_02.jpg" alt="Latent Color Guidance alone, Sintel — warm interior" width="100%"></a> | <a href="frames/GF_girl_ref-sintel_warm_03.jpg"><img src="frames/GF_girl_ref-sintel_warm_03.jpg" alt="Match + Guidance + Correction (template 08), Sintel — warm interior" width="100%"></a> |
| **Spring — cold night** | <a href="frames/frame_spring_cold.jpg"><img src="frames/frame_spring_cold.jpg" alt="Colour source, Spring — cold night" width="100%"></a> | <a href="frames/GF_girl_ref-spring_cold_02.jpg"><img src="frames/GF_girl_ref-spring_cold_02.jpg" alt="Latent Color Guidance alone, Spring — cold night" width="100%"></a> | <a href="frames/GF_girl_ref-spring_cold_03.jpg"><img src="frames/GF_girl_ref-spring_cold_03.jpg" alt="Match + Guidance + Correction (template 08), Spring — cold night" width="100%"></a> |
| **Tears of Steel — golden light** | <a href="frames/frame_tos_golden.jpg"><img src="frames/frame_tos_golden.jpg" alt="Colour source, Tears of Steel — golden light" width="100%"></a> | <a href="frames/GF_girl_ref-tos_golden_02.jpg"><img src="frames/GF_girl_ref-tos_golden_02.jpg" alt="Latent Color Guidance alone, Tears of Steel — golden light" width="100%"></a> | <a href="frames/GF_girl_ref-tos_golden_03.jpg"><img src="frames/GF_girl_ref-tos_golden_03.jpg" alt="Match + Guidance + Correction (template 08), Tears of Steel — golden light" width="100%"></a> |
| **Tears of Steel — green and lamplight** | <a href="frames/frame_tos_green.jpg"><img src="frames/frame_tos_green.jpg" alt="Colour source, Tears of Steel — green and lamplight" width="100%"></a> | <a href="frames/GF_girl_ref-tos_green_02.jpg"><img src="frames/GF_girl_ref-tos_green_02.jpg" alt="Latent Color Guidance alone, Tears of Steel — green and lamplight" width="100%"></a> | <a href="frames/GF_girl_ref-tos_green_03.jpg"><img src="frames/GF_girl_ref-tos_green_03.jpg" alt="Match + Guidance + Correction (template 08), Tears of Steel — green and lamplight" width="100%"></a> |
| **Elephants Dream — bleached** | <a href="frames/frame_ed_bleach.jpg"><img src="frames/frame_ed_bleach.jpg" alt="Colour source, Elephants Dream — bleached" width="100%"></a> | <a href="frames/GF_girl_ref-ed_bleach_02.jpg"><img src="frames/GF_girl_ref-ed_bleach_02.jpg" alt="Latent Color Guidance alone, Elephants Dream — bleached" width="100%"></a> | <a href="frames/GF_girl_ref-ed_bleach_03.jpg"><img src="frames/GF_girl_ref-ed_bleach_03.jpg" alt="Match + Guidance + Correction (template 08), Elephants Dream — bleached" width="100%"></a> |
| **The Toll of the Sea (1922) — two-strip Technicolor** | <a href="frames/frame_toll_twostrip.jpg"><img src="frames/frame_toll_twostrip.jpg" alt="Colour source, The Toll of the Sea (1922) — two-strip Technicolor" width="100%"></a> | <a href="frames/GF_girl_ref-toll_twostrip_02.jpg"><img src="frames/GF_girl_ref-toll_twostrip_02.jpg" alt="Latent Color Guidance alone, The Toll of the Sea (1922) — two-strip Technicolor" width="100%"></a> | <a href="frames/GF_girl_ref-toll_twostrip_03.jpg"><img src="frames/GF_girl_ref-toll_twostrip_03.jpg" alt="Match + Guidance + Correction (template 08), The Toll of the Sea (1922) — two-strip Technicolor" width="100%"></a> |

How far each result got from the ungraded render to the frame's colour (1 = all the way): guidance alone
0.24–0.54, the three nodes together **0.79–0.99**, within 4° of the frame's hue
wherever the frame has a hue to speak of (the bleached frame has almost none). At denoise 0.5 guidance alone is gentle:
it holds a palette well (section 3) but moves one only part of the way.

## 3. Palette plus your own wheels

A colour source sets the palette; the circles take it further. Template 09 starts the run in the source's palette with
**Latent Color Match**, and **Grade Wheels (latent)** — circles moved — takes its bearings from the model's first
prediction, which is already in that palette, holds it while the model draws and adds the adjustment on top. A second
sampler renders the same seed with the palette alone, so you see exactly what the circles add.

<a href="frames/ui_12_run_palette_plus_wheels.jpg"><img src="frames/ui_12_run_palette_plus_wheels.jpg" alt="Template 09 run: a Tears of Steel palette plus moved circles" width="100%"></a>

<p align="center"><img src="media/wheels_on_palette.jpg" alt="Four colour sources, each with the palette alone, warm circles and cool circles" width="100%"></p>

| | Colour source | Palette only | + warm circles | + cool circles |
|---|---|---|---|---|
| **Spring** | <a href="frames/frame_spring_cold.jpg"><img src="frames/frame_spring_cold.jpg" alt="Colour source, Spring" width="100%"></a> | <a href="frames/WS_girl_ref-spring_cold_match.jpg"><img src="frames/WS_girl_ref-spring_cold_match.jpg" alt="Palette only, Spring" width="100%"></a> | <a href="frames/WS_girl_ref-spring_cold_warm.jpg"><img src="frames/WS_girl_ref-spring_cold_warm.jpg" alt="+ warm circles, Spring" width="100%"></a> | <a href="frames/WS_girl_ref-spring_cold_cool.jpg"><img src="frames/WS_girl_ref-spring_cold_cool.jpg" alt="+ cool circles, Spring" width="100%"></a> |
| **Tears of Steel** | <a href="frames/frame_tos_teal.jpg"><img src="frames/frame_tos_teal.jpg" alt="Colour source, Tears of Steel" width="100%"></a> | <a href="frames/WS_girl_ref-tos_teal_match.jpg"><img src="frames/WS_girl_ref-tos_teal_match.jpg" alt="Palette only, Tears of Steel" width="100%"></a> | <a href="frames/WS_girl_ref-tos_teal_warm.jpg"><img src="frames/WS_girl_ref-tos_teal_warm.jpg" alt="+ warm circles, Tears of Steel" width="100%"></a> | <a href="frames/WS_girl_ref-tos_teal_cool.jpg"><img src="frames/WS_girl_ref-tos_teal_cool.jpg" alt="+ cool circles, Tears of Steel" width="100%"></a> |
| **Sintel** | <a href="frames/frame_sintel_warm.jpg"><img src="frames/frame_sintel_warm.jpg" alt="Colour source, Sintel" width="100%"></a> | <a href="frames/WS_girl_ref-sintel_warm_match.jpg"><img src="frames/WS_girl_ref-sintel_warm_match.jpg" alt="Palette only, Sintel" width="100%"></a> | <a href="frames/WS_girl_ref-sintel_warm_warm.jpg"><img src="frames/WS_girl_ref-sintel_warm_warm.jpg" alt="+ warm circles, Sintel" width="100%"></a> | <a href="frames/WS_girl_ref-sintel_warm_cool.jpg"><img src="frames/WS_girl_ref-sintel_warm_cool.jpg" alt="+ cool circles, Sintel" width="100%"></a> |
| **The Toll of the Sea (1922)** | <a href="frames/frame_toll_twostrip.jpg"><img src="frames/frame_toll_twostrip.jpg" alt="Colour source, The Toll of the Sea (1922)" width="100%"></a> | <a href="frames/WS_girl_ref-toll_twostrip_match.jpg"><img src="frames/WS_girl_ref-toll_twostrip_match.jpg" alt="Palette only, The Toll of the Sea (1922)" width="100%"></a> | <a href="frames/WS_girl_ref-toll_twostrip_warm.jpg"><img src="frames/WS_girl_ref-toll_twostrip_warm.jpg" alt="+ warm circles, The Toll of the Sea (1922)" width="100%"></a> | <a href="frames/WS_girl_ref-toll_twostrip_cool.jpg"><img src="frames/WS_girl_ref-toll_twostrip_cool.jpg" alt="+ cool circles, The Toll of the Sea (1922)" width="100%"></a> |

*Warm*: lift −0.03 / 0 / +0.06, gain +0.09 / +0.03 / −0.08, saturation 1.15, contrast 1.05 — the circles template 09
ships with. *Cool*: lift blue +0.03, gamma 1.12, gain −0.05 / 0 / +0.07, saturation 0.8, contrast 0.95.

Measured against the same circles laid over the palette-only render afterwards, as colourfulness against the
palette-only render: on Tears of Steel, Sintel and The Toll of the Sea the warm circles give ×1.25–1.38
(afterwards ×1.21–1.30), the cool ones ×0.60–0.68 (afterwards
×0.66–0.76). On the darkest frame, Spring's night, the warm circles barely show (×1.02
against ×1.30 afterwards) — for a dark palette, finish the warmth on the image wheels. The same
holds on three other pictures (section 10). Adding Latent Color Guidance to this chain took
38 % and 20 % of the shift back, which is why template 09 leaves it out.

## 4. Colour held through img2img

The everyday job: an img2img, refine or upscale pass that must not change the colours. The robot, prompt describing
it, seed 0, dpmpp_2m / karras, 24 steps, cfg 7, the input itself as the reference.

| | Input | img2img, nothing | A1111 Color Correction | Latent Color Guidance |
|---|---|---|---|---|
| **denoise 0.5** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, denoise 0.5" width="100%"></a> | <a href="frames/img2img_plain_d0.5.jpg"><img src="frames/img2img_plain_d0.5.jpg" alt="img2img, nothing, denoise 0.5" width="100%"></a> | <a href="frames/S_robot_d0.5_01cc.jpg"><img src="frames/S_robot_d0.5_01cc.jpg" alt="A1111 Color Correction, denoise 0.5" width="100%"></a> | <a href="frames/02_guidance_d0.5.jpg"><img src="frames/02_guidance_d0.5.jpg" alt="Latent Color Guidance, denoise 0.5" width="100%"></a> |
| **denoise 0.7** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, denoise 0.7" width="100%"></a> | <a href="frames/img2img_plain_d0.7.jpg"><img src="frames/img2img_plain_d0.7.jpg" alt="img2img, nothing, denoise 0.7" width="100%"></a> | <a href="frames/S_robot_d0.7_01cc.jpg"><img src="frames/S_robot_d0.7_01cc.jpg" alt="A1111 Color Correction, denoise 0.7" width="100%"></a> | <a href="frames/02_guidance_d0.7.jpg"><img src="frames/02_guidance_d0.7.jpg" alt="Latent Color Guidance, denoise 0.7" width="100%"></a> |
| **denoise 0.9** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, denoise 0.9" width="100%"></a> | <a href="frames/img2img_plain_d0.9.jpg"><img src="frames/img2img_plain_d0.9.jpg" alt="img2img, nothing, denoise 0.9" width="100%"></a> | <a href="frames/S_robot_d0.9_01cc.jpg"><img src="frames/S_robot_d0.9_01cc.jpg" alt="A1111 Color Correction, denoise 0.9" width="100%"></a> | <a href="frames/02_guidance_d0.9.jpg"><img src="frames/02_guidance_d0.9.jpg" alt="Latent Color Guidance, denoise 0.9" width="100%"></a> |

**A1111 Color Correction** (strength 1.0, blend `a1111`) is bit-identical to the webui. **Latent Color Guidance**
(strength 0.6, window 0.15–0.9) holds the colour while the picture is made; its strength is monotonic — LAB distance to
the input 3.11 without, 2.02 / 1.78 / 1.34 at 0.4 / 0.6 / 0.8 (denoise 0.7). On the girl at denoise 0.7: distance 5.56
without, 0.68 with the correction; guidance 5.65 → 2.24, detail within 3 %.

## 5. Loopback: eight img2img passes

**A1111 Loopback**, denoise 0.75 → 0.5 linear, seed +1 per pass. The correction takes its colour from the first frame,
as in the webui; `hold_exposure` is this pack's own addition and holds brightness too.

| | No correction | Colour correction (as A1111) | Correction + hold_exposure 0.7 |
|---|---|---|---|
| **pass 1** | <a href="frames/H_girl_cc0_pass01.jpg"><img src="frames/H_girl_cc0_pass01.jpg" alt="No correction, pass 1" width="100%"></a> | <a href="frames/H_girl_cc1_pass01.jpg"><img src="frames/H_girl_cc1_pass01.jpg" alt="Colour correction (as A1111), pass 1" width="100%"></a> | <a href="frames/H_girl_cc1-hold0.7_pass01.jpg"><img src="frames/H_girl_cc1-hold0.7_pass01.jpg" alt="Correction + hold_exposure 0.7, pass 1" width="100%"></a> |
| **pass 4** | <a href="frames/H_girl_cc0_pass04.jpg"><img src="frames/H_girl_cc0_pass04.jpg" alt="No correction, pass 4" width="100%"></a> | <a href="frames/H_girl_cc1_pass04.jpg"><img src="frames/H_girl_cc1_pass04.jpg" alt="Colour correction (as A1111), pass 4" width="100%"></a> | <a href="frames/H_girl_cc1-hold0.7_pass04.jpg"><img src="frames/H_girl_cc1-hold0.7_pass04.jpg" alt="Correction + hold_exposure 0.7, pass 4" width="100%"></a> |
| **pass 8** | <a href="frames/H_girl_cc0_pass08.jpg"><img src="frames/H_girl_cc0_pass08.jpg" alt="No correction, pass 8" width="100%"></a> | <a href="frames/H_girl_cc1_pass08.jpg"><img src="frames/H_girl_cc1_pass08.jpg" alt="Colour correction (as A1111), pass 8" width="100%"></a> | <a href="frames/H_girl_cc1-hold0.7_pass08.jpg"><img src="frames/H_girl_cc1-hold0.7_pass08.jpg" alt="Correction + hold_exposure 0.7, pass 8" width="100%"></a> |

Colour distance to the input over eight passes: 6.1–7.7 without correction, **0.3–1.3** with it. Brightness drifts
+11 → +19 L with or without the correction (the webui behaves the same); with `hold_exposure` 0.7 it stays within
+2.4…+3.7 — exactly 30 % of the drift, as the setting says.

## 6. Grade wheels, presets and LUTs

**Lift, gamma and gain** as three colour wheels — shadows, midtones, highlights — each with its master slider and its
numbers underneath; saturation and contrast below. Drag a wheel, Shift for fine moves, Alt-click to reset. Here the
*Teal and orange* preset: lift pushed to teal, gain to orange.

<a href="frames/ui_01_wheels_teal_and_orange.jpg"><img src="frames/ui_01_wheels_teal_and_orange.jpg" alt="Grade Wheels (image) with the Teal and orange preset" width="100%"></a>

**Presets, one right-click away.** 15 built-in looks plus your own: *Save these settings as a preset…* stores the
current wheels by name, *Delete a preset* removes one, *Show the numbers* opens every value as a slider.

| Right-click → Presets | Show the numbers |
|---|---|
| <a href="frames/ui_03_presets_menu.jpg"><img src="frames/ui_03_presets_menu.jpg" alt="The Presets menu next to the wheels" width="100%"></a> | <a href="frames/ui_02_wheels_numbers_shown.jpg"><img src="frames/ui_02_wheels_numbers_shown.jpg" alt="The wheels with every value as a slider" width="100%"></a> |

**Template 05** — a picture in, the wheels, before and after. No model; it runs in a fraction of a second.

<a href="frames/ui_04_workflow_grade_an_image.jpg"><img src="frames/ui_04_workflow_grade_an_image.jpg" alt="Template 05, run: before and after" width="100%"></a>

Six of the 15 built-in presets:

| | Girl | Robot |
|---|---|---|
| **input** | <a href="frames/girl.jpg"><img src="frames/girl.jpg" alt="Girl, input" width="100%"></a> | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Robot, input" width="100%"></a> |
| **Teal and orange** | <a href="frames/preset_girl_teal_and_orange.jpg"><img src="frames/preset_girl_teal_and_orange.jpg" alt="Girl, Teal and orange" width="100%"></a> | <a href="frames/preset_robot_teal_and_orange.jpg"><img src="frames/preset_robot_teal_and_orange.jpg" alt="Robot, Teal and orange" width="100%"></a> |
| **Golden hour** | <a href="frames/preset_girl_golden_hour.jpg"><img src="frames/preset_girl_golden_hour.jpg" alt="Girl, Golden hour" width="100%"></a> | <a href="frames/preset_robot_golden_hour.jpg"><img src="frames/preset_robot_golden_hour.jpg" alt="Robot, Golden hour" width="100%"></a> |
| **Moonlight** | <a href="frames/preset_girl_moonlight.jpg"><img src="frames/preset_girl_moonlight.jpg" alt="Girl, Moonlight" width="100%"></a> | <a href="frames/preset_robot_moonlight.jpg"><img src="frames/preset_robot_moonlight.jpg" alt="Robot, Moonlight" width="100%"></a> |
| **Bleach bypass** | <a href="frames/preset_girl_bleach_bypass.jpg"><img src="frames/preset_girl_bleach_bypass.jpg" alt="Girl, Bleach bypass" width="100%"></a> | <a href="frames/preset_robot_bleach_bypass.jpg"><img src="frames/preset_robot_bleach_bypass.jpg" alt="Robot, Bleach bypass" width="100%"></a> |
| **Two strip** | <a href="frames/preset_girl_two_strip.jpg"><img src="frames/preset_girl_two_strip.jpg" alt="Girl, Two strip" width="100%"></a> | <a href="frames/preset_robot_two_strip.jpg"><img src="frames/preset_robot_two_strip.jpg" alt="Robot, Two strip" width="100%"></a> |
| **Day for night** | <a href="frames/preset_girl_day_for_night.jpg"><img src="frames/preset_girl_day_for_night.jpg" alt="Girl, Day for night" width="100%"></a> | <a href="frames/preset_robot_day_for_night.jpg"><img src="frames/preset_robot_day_for_night.jpg" alt="Robot, Day for night" width="100%"></a> |

Every preset also ships as a `.cube` LUT for **Apply LUT** — and for Resolve or Premiere: preset and LUT of the same
name differ by at most 0.5 of 255 on average. **Save Grade as LUT** writes your own wheels as a 33³ cube that
round-trips within 3 of 255 (mean 0.06).

| Scopes — vectorscope, waveform, histogram (robot after Teal and orange) | Compare — wipe |
|---|---|
| <a href="frames/scopes_teal.jpg"><img src="frames/scopes_teal.jpg" alt="Scopes: vectorscope, parade, histogram" width="100%"></a> | <a href="frames/compare_wipe.jpg"><img src="frames/compare_wipe.jpg" alt="Compare: wipe" width="100%"></a> |

## 7. Other model families

The same nodes on other models, unchanged settings.

| | No grade | Wheels after | Grade Wheels (latent) |
|---|---|---|---|
| **SDXL, a dusk street** | <a href="frames/I_street_s0_none.jpg"><img src="frames/I_street_s0_none.jpg" alt="No grade, SDXL, a dusk street" width="100%"></a> | <a href="frames/I_street_s0_teal_and_orange_after.jpg"><img src="frames/I_street_s0_teal_and_orange_after.jpg" alt="Wheels after, SDXL, a dusk street" width="100%"></a> | <a href="frames/I_street_s0_teal_and_orange_while_k1.0_start0.6.jpg"><img src="frames/I_street_s0_teal_and_orange_while_k1.0_start0.6.jpg" alt="Grade Wheels (latent), SDXL, a dusk street" width="100%"></a> |
| **SD 3.5 medium** | <a href="frames/K_sd35_street_none.jpg"><img src="frames/K_sd35_street_none.jpg" alt="No grade, SD 3.5 medium" width="100%"></a> | <a href="frames/K_sd35_street_tao_after.jpg"><img src="frames/K_sd35_street_tao_after.jpg" alt="Wheels after, SD 3.5 medium" width="100%"></a> | <a href="frames/K_sd35_street_tao_while.jpg"><img src="frames/K_sd35_street_tao_while.jpg" alt="Grade Wheels (latent), SD 3.5 medium" width="100%"></a> |

Teal and orange from scratch (seed 0, 26 steps, cfg 6.5): SDXL within 3.8° of the image grade at ×0.79 of its size,
SD 3.5 within 8.3° at ×0.91 — through the decoders' measured axes (the preview matrix had given 24.5° on SD 3.5);
SD 1.5, on the preview matrix, within 4.4° at ×0.97.

| | Input | img2img, no guidance | Latent Color Guidance |
|---|---|---|---|
| **SD 1.5 (512²)** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, SD 1.5 (512²)" width="100%"></a> | <a href="frames/K_sd15_robot_d0.7_s0_plain.jpg"><img src="frames/K_sd15_robot_d0.7_s0_plain.jpg" alt="img2img, no guidance, SD 1.5 (512²)" width="100%"></a> | <a href="frames/K_sd15_robot_d0.7_s0_02.jpg"><img src="frames/K_sd15_robot_d0.7_s0_02.jpg" alt="Latent Color Guidance, SD 1.5 (512²)" width="100%"></a> |
| **SD 3.5 medium** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, SD 3.5 medium" width="100%"></a> | <a href="frames/K_sd35_robot_d0.7_s0_plain.jpg"><img src="frames/K_sd35_robot_d0.7_s0_plain.jpg" alt="img2img, no guidance, SD 3.5 medium" width="100%"></a> | <a href="frames/K_sd35_robot_d0.7_s0_02.jpg"><img src="frames/K_sd35_robot_d0.7_s0_02.jpg" alt="Latent Color Guidance, SD 3.5 medium" width="100%"></a> |

LAB distance to the input at denoise 0.7: SD 1.5 3.62 → 1.12, SD 3.5 1.48 → 0.68, detail within 3 %.

## 8. IP-Adapter and colour held

Template 07: IP-Adapter PLUS (weight 0.8) for the style, Latent Color Guidance (0.6) for the palette, one seed.

| | Film frame | IP-Adapter only | IP-Adapter + Latent Color Guidance |
|---|---|---|---|
| **Tears of Steel** | <a href="frames/frame_tos_teal.jpg"><img src="frames/frame_tos_teal.jpg" alt="Film frame, Tears of Steel" width="100%"></a> | <a href="frames/J_tos_teal_s777000_ip_only.jpg"><img src="frames/J_tos_teal_s777000_ip_only.jpg" alt="IP-Adapter only, Tears of Steel" width="100%"></a> | <a href="frames/J_tos_teal_s777000_ip_held.jpg"><img src="frames/J_tos_teal_s777000_ip_held.jpg" alt="IP-Adapter + Latent Color Guidance, Tears of Steel" width="100%"></a> |
| **Sintel** | <a href="frames/frame_sintel_warm.jpg"><img src="frames/frame_sintel_warm.jpg" alt="Film frame, Sintel" width="100%"></a> | <a href="frames/J_sintel_warm_s777000_ip_only.jpg"><img src="frames/J_sintel_warm_s777000_ip_only.jpg" alt="IP-Adapter only, Sintel" width="100%"></a> | <a href="frames/J_sintel_warm_s777000_ip_held.jpg"><img src="frames/J_sintel_warm_s777000_ip_held.jpg" alt="IP-Adapter + Latent Color Guidance, Sintel" width="100%"></a> |

IP-Adapter already carries most of a palette; the guidance brings the colour 17 % closer to the frame on average
(×0.77–0.91 over five cases), detail within 4 %.

## 9. Masks

| | Mask (red: sky) | Correction everywhere | Correction inside the mask |
|---|---|---|---|
| **A1111 Color Correction** | <a href="frames/mask_overlay.jpg"><img src="frames/mask_overlay.jpg" alt="Mask (red: sky), A1111 Color Correction" width="100%"></a> | <a href="frames/M_robot_plain_d0.7_01cc_ref-tos_teal_full.jpg"><img src="frames/M_robot_plain_d0.7_01cc_ref-tos_teal_full.jpg" alt="Correction everywhere, A1111 Color Correction" width="100%"></a> | <a href="frames/M_robot_plain_d0.7_01cc_ref-tos_teal_mask-no-sky.jpg"><img src="frames/M_robot_plain_d0.7_01cc_ref-tos_teal_mask-no-sky.jpg" alt="Correction inside the mask, A1111 Color Correction" width="100%"></a> |

Outside a mask the image nodes change no pixel (max |Δ| 0); inside, the correction measures its histograms there. The
latent nodes take a mask too: it limits where the colour is *measured*, the usual want for inpainting.

## 10. Checked on three more pictures

Everything above was measured on the girl first. The same tests on three more pictures — a photo-style robot, a
photo-style forest chase at 1200×680 and a painted crowd — SDXL img2img at denoise 0.5, seed 0, the same settings.

**Grade wheels inside sampling** (Golden hour):

| | Input | No grade | Wheels after | Wheels inside sampling |
|---|---|---|---|---|
| **Robot** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, Robot" width="100%"></a> | <a href="frames/RC_robot_none.jpg"><img src="frames/RC_robot_none.jpg" alt="No grade, Robot" width="100%"></a> | <a href="frames/RC_robot_golden_hour_after.jpg"><img src="frames/RC_robot_golden_hour_after.jpg" alt="Wheels after, Robot" width="100%"></a> | <a href="frames/RC_robot_golden_hour_during_k1.0.jpg"><img src="frames/RC_robot_golden_hour_during_k1.0.jpg" alt="Wheels inside sampling, Robot" width="100%"></a> |
| **Forest chase** | <a href="frames/truck.jpg"><img src="frames/truck.jpg" alt="Input, Forest chase" width="100%"></a> | <a href="frames/RC_truck_none.jpg"><img src="frames/RC_truck_none.jpg" alt="No grade, Forest chase" width="100%"></a> | <a href="frames/RC_truck_golden_hour_after.jpg"><img src="frames/RC_truck_golden_hour_after.jpg" alt="Wheels after, Forest chase" width="100%"></a> | <a href="frames/RC_truck_golden_hour_during_k1.0.jpg"><img src="frames/RC_truck_golden_hour_during_k1.0.jpg" alt="Wheels inside sampling, Forest chase" width="100%"></a> |
| **Witch's selfie** | <a href="frames/witch.jpg"><img src="frames/witch.jpg" alt="Input, Witch's selfie" width="100%"></a> | <a href="frames/RC_witch_none.jpg"><img src="frames/RC_witch_none.jpg" alt="No grade, Witch's selfie" width="100%"></a> | <a href="frames/RC_witch_golden_hour_after.jpg"><img src="frames/RC_witch_golden_hour_after.jpg" alt="Wheels after, Witch's selfie" width="100%"></a> | <a href="frames/RC_witch_golden_hour_during_k1.0.jpg"><img src="frames/RC_witch_golden_hour_during_k1.0.jpg" alt="Wheels inside sampling, Witch's selfie" width="100%"></a> |

Distance between the latent result and the same wheels applied afterwards, over Teal and orange, Golden hour and
Moonlight, in LAB units: robot 2.2–3.6, witch 2.4–2.8 — and forest chase 4.8–6.4, the one scene
where the latent wheels miss: dark, green and snowy, Golden hour there comes out greener rather than golden and
Moonlight weaker. The code before the current one missed it too (3.8–5.6): the scene is the cause, and the
current code is slightly further off there, while it is closer on the robot, the witch and the girl.
Fine detail inside sampling stays at ×0.87–1.04 of the ungraded render (the code before: ×0.72–1.37).
`strength` 0.5 gives ×0.58–0.78 of the full shift on these pictures.

**A picture's palette** (template 08):

| | Input | Spring | Sintel | The Toll of the Sea |
|---|---|---|---|---|
| **Robot** | <a href="frames/robot.jpg"><img src="frames/robot.jpg" alt="Input, Robot" width="100%"></a> | <a href="frames/RC_robot_ref-spring_cold_08.jpg"><img src="frames/RC_robot_ref-spring_cold_08.jpg" alt="Spring, Robot" width="100%"></a> | <a href="frames/RC_robot_ref-sintel_warm_08.jpg"><img src="frames/RC_robot_ref-sintel_warm_08.jpg" alt="Sintel, Robot" width="100%"></a> | <a href="frames/RC_robot_ref-toll_twostrip_08.jpg"><img src="frames/RC_robot_ref-toll_twostrip_08.jpg" alt="The Toll of the Sea, Robot" width="100%"></a> |
| **Forest chase** | <a href="frames/truck.jpg"><img src="frames/truck.jpg" alt="Input, Forest chase" width="100%"></a> | <a href="frames/RC_truck_ref-spring_cold_08.jpg"><img src="frames/RC_truck_ref-spring_cold_08.jpg" alt="Spring, Forest chase" width="100%"></a> | <a href="frames/RC_truck_ref-sintel_warm_08.jpg"><img src="frames/RC_truck_ref-sintel_warm_08.jpg" alt="Sintel, Forest chase" width="100%"></a> | <a href="frames/RC_truck_ref-toll_twostrip_08.jpg"><img src="frames/RC_truck_ref-toll_twostrip_08.jpg" alt="The Toll of the Sea, Forest chase" width="100%"></a> |
| **Witch's selfie** | <a href="frames/witch.jpg"><img src="frames/witch.jpg" alt="Input, Witch's selfie" width="100%"></a> | <a href="frames/RC_witch_ref-spring_cold_08.jpg"><img src="frames/RC_witch_ref-spring_cold_08.jpg" alt="Spring, Witch's selfie" width="100%"></a> | <a href="frames/RC_witch_ref-sintel_warm_08.jpg"><img src="frames/RC_witch_ref-sintel_warm_08.jpg" alt="Sintel, Witch's selfie" width="100%"></a> | <a href="frames/RC_witch_ref-toll_twostrip_08.jpg"><img src="frames/RC_witch_ref-toll_twostrip_08.jpg" alt="The Toll of the Sea, Witch's selfie" width="100%"></a> |

0.62–0.98 of the way to the source's colour on these three. On the robot the chain also rewrote the lettering
on its chest; on the girl, the forest chase and the witch it kept the picture.

## Every node, one by one

Ten nodes, under **Color Grade Tools** in the node menu; type "Color Grade" in the search to list them all. Each is shown
as it appears in the editor.

### Color Grade · Grade Wheels (latent)

<table><tr><td width="44%" valign="top"><a href="frames/node_LatentColorGrade.jpg"><img src="frames/node_LatentColorGrade.jpg" alt="Color Grade · Grade Wheels (latent)" width="100%"></a></td><td valign="top">

`LatentColorGrade` · menu *Color Grade Tools / model*

**Grades the picture while the model draws it.** The same
lift / gamma / gain wheels, saturation and contrast as the image node, applied to the model's prediction at every step
of its window, so the look is composed rather than laid over.

- **Takes** `MODEL`, optional `anchor` latent · **gives** `MODEL` — between the checkpoint and the sampler.
- **strength** 1.0 is the full look; 0.5 gives 0.39–0.78 of it, depending on the picture.
- **start_percent** 0.6 / **end_percent** 1.0 — the window. It holds the levels of its first prediction, so it also
  resists what the model repaints later; from 0.3 an img2img at denoise 0.7 came out 3.4 times the grade, from 0.6 1.1.
- **anchor** — aim at a latent's levels and colourfulness instead of the model's own: the img2img source, or a picture
  whose palette you want (gentler than template 08, see [what it does not promise](#what-it-does-not-promise)).
- Right-click: *Presets* (15 built in + yours), *Save these settings as a preset…*, *Show the numbers*.
- Templates **06**, **09**. Sections [1](#1-grade-wheels-inside-sampling), [3](#3-palette-plus-your-own-wheels).

</td></tr></table>

### Color Grade · Latent Color Guidance

<table><tr><td width="44%" valign="top"><a href="frames/node_ApplyLatentColorGuidance.jpg"><img src="frames/node_ApplyLatentColorGuidance.jpg" alt="Color Grade · Latent Color Guidance" width="100%"></a></td><td valign="top">

`ApplyLatentColorGuidance` · menu *Color Grade Tools / model*

**Holds a reference's palette through
sampling.** At every model call inside its window the prediction is moved towards the reference's colour statistics.

- **Takes** `MODEL` + `reference_latent` (VAE Encode of the picture to match), optional `mask` · **gives** `MODEL`.
- **strength** 0.5–0.7 per model call · **start_percent** 0.15 (the first predictions are featureless) ·
  **end_percent** 0.9 (the last steps settle detail).
- **method** `auto` (`reinhard` on 4-channel latents, `mean_only` where channels are not colour axes), `mkl` for casts
  lying across channels · **space** `auto` = the RGB projection that works on every family · **max_gain**, **lowpass**
  against amplified detail · **temporal** for video: one correction per clip, or per frame.
- Use it for img2img, refine and upscale passes from denoise ≈ 0.5 up; below ≈ 0.3 it warns and does nothing.
- Templates **02**, **03**, **08**. Sections [2](#2-any-picture-becomes-a-palette), [4](#4-colour-held-through-img2img).

</td></tr></table>

### Color Grade · Latent Color Match

<table><tr><td width="44%" valign="top"><a href="frames/node_LatentColorMatch.jpg"><img src="frames/node_LatentColorMatch.jpg" alt="Color Grade · Latent Color Match" width="100%"></a></td><td valign="top">

`LatentColorMatch` · menu *Color Grade Tools / latent*

**One-shot colour match of a latent** to a reference
latent — before sampling, to start from the right palette, or after it.

- **Takes** `samples` + `reference`, optional `model` (recommended: it gives the RGB projection) and `mask` ·
  **gives** `LATENT`.
- **strength**, **method**, **space**, **max_gain** as in the guidance.
- Templates **03**, **08**, **09** — on the init latent, so the run starts in the source's palette.

</td></tr></table>

### Color Grade · A1111 Color Correction

<table><tr><td width="44%" valign="top"><a href="frames/node_A1111ColorCorrection.jpg"><img src="frames/node_A1111ColorCorrection.jpg" alt="Color Grade · A1111 Color Correction" width="100%"></a></td><td valign="top">

`A1111ColorCorrection` · menu *Color Grade Tools / image*

**The webui's colour correction, bit for bit.**
Histogram match in LAB, then a luminosity blend: colour from the reference, brightness from your render.

- **Takes** `image` + `reference`, optional `mask` · **gives** `IMAGE`.
- **strength** 0–1 · **blend_mode** `a1111` (the webui), `lab_color`, `lab_luminosity` (the old inverted behaviour,
  honestly labelled), `none` · **match_channels** `L+a+b` or `a+b` (chroma only) · **ref_blur** to take only a
  reference's overall cast (not in A1111).
- Exact: 0 of 255 against the webui on its pinned libraries.
- Templates **01**, **03**, **08**. Sections [4](#4-colour-held-through-img2img), [9](#9-masks).

</td></tr></table>

### Color Grade · A1111 Loopback

<table><tr><td width="44%" valign="top"><a href="frames/node_A1111Loopback.jpg"><img src="frames/node_A1111Loopback.jpg" alt="Color Grade · A1111 Loopback" width="100%"></a></td><td valign="top">

`A1111Loopback` · menu *Color Grade Tools / sampling*

**The webui's Loopback script**: a chain of img2img passes,
each output the next input, every pass corrected against the first frame's colours.

- **Takes** `model`, `vae`, `positive`, `negative`, `image`, optional `mask` · **gives** `all_passes`, `final`.
- **loops**, **denoise** → **final_denoise** along a **denoise_curve** (Linear / Aggressive / Lazy), seed +1 per pass,
  **color_correction** on/off.
- **hold_exposure** — this pack's addition: also pulls brightness back, which A1111's blend leaves free.
- **mask** — sample only inside; the rest is pasted back from the original after each pass.
- Template **04**. Section [5](#5-loopback-eight-img2img-passes).

</td></tr></table>

### Color Grade · Grade Wheels (image)

<table><tr><td width="44%" valign="top"><a href="frames/node_ImageColorGrade.jpg"><img src="frames/node_ImageColorGrade.jpg" alt="Color Grade · Grade Wheels (image)" width="100%"></a></td><td valign="top">

`ImageColorGrade` · menu *Color Grade Tools / image*

**Grades a finished picture** — exact, instant, no model.

- **Takes** `image`, optional `reference` and `mask` · **gives** `IMAGE`.
- Lift / gamma / gain wheels with master sliders, **saturation**, **contrast**; the same presets menu.
- **reference** — the wheels start from another picture's look: the image is first mapped onto its shadows, midtones
  and highlights.
- Template **05**. Section [6](#6-grade-wheels-presets-and-luts).

</td></tr></table>

### Color Grade · Apply LUT

<table><tr><td width="44%" valign="top"><a href="frames/node_ColorKitApplyLUT.jpg"><img src="frames/node_ColorKitApplyLUT.jpg" alt="Color Grade · Apply LUT" width="100%"></a></td><td valign="top">

`ColorKitApplyLUT` · menu *Color Grade Tools / image*

**Any `.cube` LUT** from `models/luts` — film emulation, print
stock, your own. The 14 built-in looks ship as `.cube` files too.

- **Takes** `image` · **gives** `IMAGE` · **lut_name**, **strength** (most film LUTs are built for log footage; half
  strength is often enough on an ordinary image).

</td></tr></table>

### Color Grade · Save Grade as LUT

<table><tr><td width="44%" valign="top"><a href="frames/node_ColorKitSaveGradeAsLUT.jpg"><img src="frames/node_ColorKitSaveGradeAsLUT.jpg" alt="Color Grade · Save Grade as LUT" width="100%"></a></td><td valign="top">

`ColorKitSaveGradeAsLUT` · menu *Color Grade Tools / image*

**Exports the wheels as a `.cube`** for Resolve,
Premiere or another pack.

- Wheels, **filename** (into `models/luts`), **size** 33 points per axis; optional `image` to preview the grade ·
  **gives** `image`, `path`.
- Round-trips within 3 of 255 (mean 0.06).

</td></tr></table>

### Color Grade · Scopes

<table><tr><td width="44%" valign="top"><a href="frames/node_ColorKitScopes.jpg"><img src="frames/node_ColorKitScopes.jpg" alt="Color Grade · Scopes" width="100%"></a></td><td valign="top">

`ColorKitScopes` · menu *Color Grade Tools / inspect*

**Vectorscope, waveform, RGB parade, histogram** of an image — or of a
latent, projected through the model's own matrix (approximate, like the previews).

- **Takes** `image`, or `latent` + `model` · **gives** the scopes as `IMAGE` and a `clipping` report.
- **mode** all / vectorscope / waveform / parade / histogram · **size** of a panel.

</td></tr></table>

### Color Grade · Compare

<table><tr><td width="44%" valign="top"><a href="frames/node_ColorKitCompare.jpg"><img src="frames/node_ColorKitCompare.jpg" alt="Color Grade · Compare" width="100%"></a></td><td valign="top">

`ColorKitCompare` · menu *Color Grade Tools / inspect*

**Two pictures, one view**: wipe, side by side, amplified difference
or checkerboard.

- **Takes** `image_a`, `image_b` · **gives** `IMAGE` · **mode**, **position** of the wipe, **amplify** for the
  difference — the honest way to see a small change.

</td></tr></table>

A deprecated **KSampler (old)** is still registered, hidden from the menu, so workflows built on the original private
nodes keep opening.

## Try it

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BenderBlender/ComfyUI-Color-Grade-Tools-A1111
```

Or in ComfyUI Manager: *Install via Git URL* with the same address. Restart ComfyUI; nothing else to install, no models
to download. The templates appear under **Workflow → Browse Templates → ComfyUI-Color-Grade-Tools-A1111**:

| Template | What it shows |
|---|---|
| 01 img2img color correction | A1111's post-pass after an img2img |
| 02 latent color guidance | the input's colour held while the model draws |
| 03 guidance plus correction | both stages — the usual refine / upscale setup |
| 04 a1111 loopback | the Loopback chain, every pass previewed |
| 05 grade an image | the wheels on a picture; no model |
| 06 grade while sampling | the latent wheels against the same wheels afterwards |
| 07 ipadapter with color grade | IP-Adapter for style, guidance for the palette |
| 08 palette from any picture | your picture in the palette of another one |
| 09 palette plus your wheels | that palette, with your own circles on top — and the palette alone beside it |

Start with **08** and two pictures of your own: change only the colour source and watch the picture follow it.
Then open **09** and move the circles.
