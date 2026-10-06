# ComfyUI-QwenTextEncoder

**Text Encode (Qwen Image 2.1)** — vision-aware text conditioning for **Qwen Image 2.1**,
whose text encoder is **Qwen3-VL-8B** (`comfy/text_encoders/qwen_image21.py`).

Reference images are pushed through the VLM **vision path** to make the conditioning
visually aware of them — a "prompt from a picture" effect. It accepts an **unbounded,
auto-growing set of image+mask pairs** (connect `image1`, a fresh `image2`/`mask2`
pair appears, and so on).

## Qwen Image 2.1 specifics (accounted for)

| Aspect | Value |
|---|---|
| Text encoder | Qwen3-VL-8B (`model_type="qwen3vl_8b"`, vision depth 27, patch 16, merge 2, DeepStack [8, 16, 24]) |
| CLIPLoader | type **`qwen_image`** |
| System instruction | `Comprehend and analyze the provided prompt.` (trained default) |
| Vision markers | `<imageN><\|vision_start\|><\|image_pad\|><\|vision_end\|>`, space-joined |
| Sizing | `resolution × resolution` at **multiples of 32**, aspect preserved, `lanczos` — the **same** pixels feed the vision tower and the VAE, so every vision slot covers whole latents. `0` = native size rounded to 32 |
| Empty latent | `[1, 64, H//16, W//16]`, sized to the first reference (else `resolution`, else 1024) |
| FP8 | An FP8-loaded encoder dies on the image path — use bf16/fp16 with images (text-only works in FP8) |

## VAE (the edit path)

Unlike Krea2 (pure T2I, no reference-latent slot), the Qwen Image 2.1 DiT
(`comfy/ldm/qwen_image21/model.py`, `build_sequence`) **splices VAE reference
latents** into its sequence at `image_slots`:

* **VAE connected + images present** → `keep_vision=False`: vision tokens are
  dropped, each image is VAE-encoded and spliced as a reference latent
  (pixel-faithful edit path).
* **VAE unconnected (or no images)** → `keep_vision=True`: the image conditions
  through the text encoder alone.

## Masks

Each reference image has an optional companion mask. When connected, the image is
**cropped to the mask's bounding box** before resizing, so the VLM only sees the
masked region. Use **`mask_padding`** to keep surrounding context: it grows the
crop box by that fraction of the image size on each side (`0` = tight crop,
`0.1` ≈ 10% margin). The mask is used only to compute the crop — it is not itself
sent to the VLM. This is *reference-image* masking — **not inpainting**. (To
spatially restrict where the *prompt* applies in the output, use ComfyUI's
standard `ConditioningSetMask` downstream of this node.)

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| `clip` | CLIP | Load with **CLIPLoader → type `qwen_image`**. |
| `prompt` | STRING | Your text prompt (positive). |
| `negative_prompt` | STRING | Negative prompt (sees the same references). |
| `vae` | VAE | Optional. Connected = edit path with reference latents. |
| `image1…N` | IMAGE | Optional reference images; slots grow as you connect them. |
| `mask1…N` | MASK | Optional per-image mask; crops `imageN` to the masked bounding box. |
| `resolution` | INT | Reference size, multiples of 32, aspect preserved (`0` = native). Default `1024`. |
| `mask_padding` | FLOAT | Context kept around the mask, as a fraction of image size per side (`0` = tight, default). |
| `system_prompt` | STRING input | Optional. Wire a text node to override the system instruction; unconnected = trained default. See below. |
| `vision_position` | CHOICE | Image tokens `before prompt` (default) or `after prompt` in the user turn. |
| `print_prompt` | BOOLEAN | Print the assembled Qwen3-VL prompt to the ComfyUI console (debug). |

**Outputs:** `positive` + `negative` conditioning and an empty `latent` sized to
the first reference (wire it into the sampler — any other size shifts the edit).

## System prompt (making the prompt interact with the image)

`system_prompt` is a connectable text **input**, not a widget — leave it unconnected to use
the trained instruction (the in-distribution default), or wire a text node to override it.
Provide just the instruction text; the node wraps it in the chat-template scaffolding.

By default the VLM only analyses the prompt next to the image. To make them **interact**
(à la Qwen-Image-Edit), feed an instruct-style instruction. The package ships a
**`Qwen 2.1 System Prompt`** node preloaded with that instruction — just drop it and
wire its output into `system_prompt` (edit the text if you like). Its default text is:

> Describe the key features of the reference image (color, shape, size, texture, objects,
> background), then explain how the user's instruction should combine with or alter it, and
> generate a new image meeting the instruction while staying consistent with the reference
> where appropriate:

This is **experimental / out-of-distribution** — results may drift. A/B it against the default.

## Install

Symlink (or copy) this repo into ComfyUI's `custom_nodes/`:

```bash
ln -s /path/to/ComfyUI-QwenTextEncoder /path/to/ComfyUI/custom_nodes/ComfyUI-QwenTextEncoder
```

Then restart ComfyUI.

## Credits

UX (auto-growing image+mask pairs, mask crop, system-prompt override) adapted from
[ComfyUI-Krea2TextEncoder](https://github.com/ethanfel/ComfyUI-Krea2TextEncoder) (MIT).
Qwen Image 2.1 behavior mirrors ComfyUI core `TextEncodeQwenImage21`
(`comfy_extras/nodes_qwen.py`).
