# Model support and working resolutions

These nodes carry no resolution rule of their own. They read whatever latent
the sampler hands them and write back the same shape, so the only sizes that
matter are the ones the *model* imposes. This page collects both: the hard
constraint, which is in ComfyUI's source and is checked here, and the
resolution a family was trained at, which is convention rather than code.

## The hard constraint: latent downscale

Every latent format declares `spacial_downscale_ratio`. Width and height must
be a multiple of it; `VAEEncode` silently centre-crops anything that is not,
so an odd input size quietly loses a few pixels rather than failing.

| Family | Latent channels | Downscale | Image size must be a multiple of |
|---|---|---|---|
| SD 1.x, SD 2.x | 4 | 8 | 8 |
| SDXL, Playground v2.5 | 4 | 8 | 8 |
| SD 3.x | 16 | 8 | 8 |
| Flux.1 | 16 | 8 | 8 |
| Flux.2 | 128 | 16 | 16 |
| Qwen-Image | 16 | 16 | 16 |
| Wan 2.1 | 16 | 8 | 8 |
| Wan 2.2 | 48 | 16 | 16 |
| HunyuanVideo | 16 | 8 | 8 |
| HunyuanVideo 1.5 | 16 | 16 | 16 |
| HunyuanImage 2.1 | 16 | 32 | 32 |
| LTX-Video | 128 | 32 | 32 |
| Stable Cascade (prior) | 16 | 42 | 42 |
| ChromaRadiance and other pixel-space models | 3 | 1 | 1 |

Taken from `comfy/latent_formats.py`; read at runtime rather than assumed, so
a family added to ComfyUI later is picked up without a change here.

## The convention: what each family was trained at

Going far from these produces duplicated limbs and repeated composition —
a property of the model, not of these nodes.

| Family | Native | Comfortable range |
|---|---|---|
| SD 1.4 / 1.5 | 512 x 512 | 512-768, with 768 already prone to duplication |
| SD 2.0 base | 512 x 512 | 512-768 |
| SD 2.0 / 2.1 "-768" | 768 x 768 | 640-896 |
| SDXL | 1024 x 1024 | the bucket list below, roughly 1 megapixel |
| SD 3.0 / 3.5 | 1024 x 1024 | 512-1536 |
| Flux.1 | 1024 x 1024 | 512-2048 |

SDXL's training buckets, all close to 1 MP and all multiples of 64:

```
1024x1024   1152x896   896x1152   1216x832   832x1216
1344x768    768x1344   1536x640   640x1536
```

## What this means for these nodes

**The reference does not have to match the size of what you are sampling.**
Only its colour statistics are read, so a 512x512 reference can steer a
1536x1536 render. That is deliberate: it lets you keep a small, clean
reference beside a large upscale pass.

**The reference does have to come from the same VAE.** A latent encoded by
SD1.5's VAE has 4 channels and one encoded by Flux's has 16; feeding the wrong
one is caught with a message naming both counts rather than failing inside the
sampler.

**Masks are resized for you**, to the latent resolution actually being
sampled, which is not always the reference's — detailers hand over small crops
mid-sample, and that case is handled per call.

**Video models** take a still image as a reference: it is given the time axis
the format expects, and its statistics then steer every frame. With
`temporal = global` one correction is computed for the whole clip, which is
what keeps it from flickering.

**Audio and 3D latent formats are refused** with an explanation. Their
channels carry no colour, so a "colour match" there would be a meaningless
operation dressed up as a useful one.
