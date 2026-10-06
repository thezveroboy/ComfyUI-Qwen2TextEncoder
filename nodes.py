"""Text Encode (Qwen Image 2.1) — vision-aware conditioning for Qwen Image 2.1.

Qwen Image 2.1 conditions on Qwen3-VL-8B (``comfy/text_encoders/qwen_image21.py``,
``model_type="qwen3vl_8b"``). Because that text encoder is a vision-language model,
a reference image can be fed through its *vision* path so the conditioning becomes
visually informed by the image — "prompt from a picture".

Unlike Krea2 (pure text-to-image DiT, no reference-latent slot), the Qwen Image 2.1
DiT (``comfy/ldm/qwen_image21/model.py``) *does* splice VAE reference latents into
its token sequence at ``image_slots`` (``build_sequence``). So this node keeps the
optional VAE input from the core ``TextEncodeQwenImage21`` node:

* VAE connected + images present → vision tokens are dropped (``keep_vision=False``),
  each image is VAE-encoded and spliced as a reference latent (pixel-faithful edit
  path). The vision tower still "sees" the image during tokenization; the encoder
  records where each image goes (``image_slots``).
* VAE unconnected (or no images) → ``keep_vision=True``: the image conditions
  through the text encoder alone, no reference latents.

On top of the core node this adds the Krea2-node UX: an unbounded, auto-growing
set of image+mask pairs, per-image mask crop (with ``mask_padding`` context),
``vision_position`` (image tokens before/after the prompt), overridable
``system_prompt``, ``print_prompt`` debug, and the same resize for the vision
tower and the VAE (multiples of 32, ``lanczos``) so every vision slot covers
whole latents.

Each reference image has an optional companion mask. When a mask is connected the
image is cropped to the mask's bounding box before anything else, so the VLM only
"sees" the masked region. (Reference-image masking; it is not inpainting.)
"""

import math
import re

import torch

import comfy.utils

# Keep in sync with the model's own constants; fall back to literal copies.
try:
    from comfy.text_encoders.qwen_image21 import (
        SYSTEM_PROMPT as _QWEN21_SYSTEM_PROMPT,
        T2I_TEMPLATE as _QWEN21_T2I_TEMPLATE,
        VISION_BLOCK as _QWEN21_VISION_BLOCK,
    )
except Exception:  # pragma: no cover - portability shim
    _QWEN21_SYSTEM_PROMPT = (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    )
    _QWEN21_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
    _QWEN21_T2I_TEMPLATE = (
        _QWEN21_SYSTEM_PROMPT
        + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )

VISION_BLOCK = _QWEN21_VISION_BLOCK
QWEN21_T2I_TEMPLATE = _QWEN21_T2I_TEMPLATE

# The user-facing system_prompt field holds just the system *message text*; the node
# wraps it in the chat-template scaffolding. Pull the default out of the template so
# it stays in sync with whatever comfy ships.
_sys = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", QWEN21_T2I_TEMPLATE, re.S)
QWEN21_SYSTEM_DEFAULT = _sys.group(1) if _sys else (
    "Comprehend and analyze the provided prompt."
)

# Instruct/edit-style framing (a la TextEncodeQwenImageEditPlus): paste this into
# system_prompt to make the VLM fuse the user's text WITH the reference image
# instead of just analysing the prompt next to it. Out-of-distribution for Qwen
# Image 2.1's trained descriptor — experimental.
QWEN21_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
)


class TextEncodeQwen21:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                # system_prompt sits just above the image slots.
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional system-instruction input. Wire a text node to override how the "
                               "VLM frames the reference + your prompt; leave unconnected to use Qwen "
                               "Image 2.1's trained instruction (in-distribution). Use an instruct/edit-style "
                               "instruction (see README) to fuse the prompt with the image. The node "
                               "adds the chat-template scaffolding; provide just the instruction text.",
                }),
                # image1/mask1 are the seed pair; the web extension grows image2/mask2, ... on connect.
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "resolution": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Reference images are resized to about resolution x resolution pixels, "
                               "at multiples of 32, preserving aspect ratio — the SAME pixels feed the "
                               "vision tower and the VAE, so every vision slot covers whole latents. "
                               "0 keeps each reference at its own size, rounded to a multiple of 32.",
                }),
                "mask_padding": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02,
                    "tooltip": "Context kept around the mask before cropping, as a fraction of the "
                               "image size added on EACH side. 0 = tight crop to the mask; 0.1 = ~10% "
                               "margin of surroundings. Only applies when a mask is connected.",
                }),
                "vision_position": (["before prompt", "after prompt"], {
                    "default": "before prompt",
                    "tooltip": "Where the image (vision) tokens sit in the user turn relative to your "
                               "text. 'before prompt' = image then text (default); 'after prompt' = text "
                               "then image. No effect without an image. Experimental.",
                }),
                "print_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the full assembled prompt sent to the Qwen3-VL encoder (system "
                               "instruction + vision placeholders + your text) to the ComfyUI console.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/qwen image"
    DESCRIPTION = ("Qwen Image 2.1 text conditioning with optional vision prompting. Reference images "
                   "are fed through the Qwen3-VL-8B vision path; an optional per-image mask crops the "
                   "image to the masked region. With a VAE connected, references are also spliced into "
                   "the DiT as reference latents (edit path); without it, the image conditions through "
                   "the text encoder alone.")

    @staticmethod
    def _collect_indexed(kwargs, prefix):
        pattern = re.compile(r"^{}(\d+)$".format(prefix))
        out = {}
        for key, value in kwargs.items():
            match = pattern.match(key)
            if match is not None and value is not None:
                out[int(match.group(1))] = value
        return out

    @staticmethod
    def _crop_to_mask(image, mask, padding=0.0):
        """Crop image (B,H,W,C) to the mask bounding box, expanded by `padding` (a
        fraction of the image size) on each side. No-op if mask empty/None."""
        if mask is None:
            return image

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:  # (B,1,H,W) or similar -> (B,H,W)
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])

        h, w = image.shape[1], image.shape[2]
        if mask.shape[-2:] != (h, w):
            resized = comfy.utils.common_upscale(mask.unsqueeze(1), w, h, "bilinear", "disabled")
            mask = resized[:, 0]

        presence = (mask > 0.5).any(dim=0)  # collapse batch -> (H,W)
        if not bool(presence.any()):
            return image  # nothing selected: keep the whole image

        rows = torch.where(torch.any(presence, dim=1))[0]
        cols = torch.where(torch.any(presence, dim=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])

        if padding > 0.0:  # grow the box outward for surrounding context, clamped to the image
            pad_x = round(padding * w)
            pad_y = round(padding * h)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)

        return image[:, y0:y1 + 1, x0:x1 + 1, :]

    @staticmethod
    def _resize_like_core(image, resolution):
        """Resize one image (1,H,W,C) like the core TextEncodeQwenImage21 node: same
        pixels for the vision tower and the VAE, multiples of 32, aspect preserved.
        Returns (resized, width, height)."""
        samples = image.movedim(-1, 1)
        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
        width, height = max(32, width), max(32, height)
        if (width, height) == (samples.shape[3], samples.shape[2]):
            return image, width, height
        s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
        return s, width, height

    @classmethod
    def _prepare_vision(cls, kwargs, resolution, mask_padding):
        """Crop+resize each connected reference. Returns (images_vl, image_prompt,
        vae_inputs, first_wh). images_vl: RGB over white (vision tower). vae_inputs:
        full resized tensors (alpha kept, like the core node)."""
        images = cls._collect_indexed(kwargs, "image")
        masks = cls._collect_indexed(kwargs, "mask")
        ordered = sorted(images.keys())

        images_vl = []
        vae_inputs = []
        refs = []
        first_wh = None

        for n in ordered:
            cropped = cls._crop_to_mask(images[n][:1], masks.get(n), padding=mask_padding)
            s, width, height = cls._resize_like_core(cropped, resolution)
            if first_wh is None:
                first_wh = (width, height)
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                # the vision tower sees alpha over white, the vae keeps all channels
                alpha = s[:, :, :, 3:]
                rgb = rgb * alpha + (1.0 - alpha)
            images_vl.append(rgb)
            vae_inputs.append(s)
            refs.append("<image{}>{}".format(len(refs) + 1, VISION_BLOCK))

        return images_vl, " ".join(refs), vae_inputs, first_wh

    @staticmethod
    def _build_text(system_prompt, prompt, image_prompt, vision_position):
        """Assemble the user text (with vision tokens) and the chat template."""
        system = (system_prompt or "").strip() or QWEN21_SYSTEM_DEFAULT
        template = ("<|im_start|>system\n" + system + "<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        if image_prompt:
            text = (prompt + image_prompt) if vision_position == "after prompt" else (image_prompt + prompt)
        else:
            text = prompt
        return text, template

    @staticmethod
    def _fp8_hint(exc, images_vl):
        """Map the cryptic FP8 vision crash to an actionable error; else None.

        ComfyUI's Qwen3-VL vision tower (qwen35.py fast_pos_embed_interpolate) adds the pos-embed
        weights without casting, so an FP8-loaded text encoder dies on the image path."""
        if images_vl and isinstance(exc, NotImplementedError) and "Float8" in str(exc):
            return RuntimeError(
                "Qwen Image 2.1: the Qwen3-VL-8B text encoder is loaded in FP8, which ComfyUI's vision "
                "tower cannot run on the image path ('add_stub not implemented for Float8_e4m3fn'). Load "
                "a bf16/fp16 Qwen3-VL-8B text encoder (CLIPLoader type 'qwen_image') when using image "
                "references. The FP8 encoder works only text-only."
            )
        return None

    def encode(self, clip, prompt, negative_prompt, vae=None, resolution=1024, mask_padding=0.0,
               system_prompt=QWEN21_SYSTEM_DEFAULT, vision_position="before prompt",
               print_prompt=False, **kwargs):
        images_vl, image_prompt, vae_inputs, first_wh = self._prepare_vision(kwargs, resolution, mask_padding)

        ref_latents = []
        if vae is not None:
            for s in vae_inputs:
                ref_latents.append(vae.encode(s))

        # With reference latents coming, vision tokens are replaced by them in the DiT
        # (image_slots); without latents the image conditions through the text encoder alone.
        keep_vision = len(ref_latents) == 0

        text_pos, template = self._build_text(system_prompt, prompt, image_prompt, vision_position)
        text_neg, _ = self._build_text(system_prompt, negative_prompt, image_prompt, vision_position)

        if print_prompt:
            print("\n========== Text Encode (Qwen21) -> Qwen3-VL prompt ==========")
            print(template.replace("{}", text_pos, 1))  # literal replace: brace-safe
            print("---- references: {} (keep_vision={}) ----".format(len(images_vl), keep_vision))
            print("============================================================\n")

        tokens_pos = clip.tokenize(text_pos, images=images_vl, llama_template=template,
                                   keep_vision=keep_vision, prevent_empty_text=True)
        tokens_neg = clip.tokenize(text_neg, images=images_vl, llama_template=template,
                                   keep_vision=keep_vision, prevent_empty_text=True)
        try:
            positive = clip.encode_from_tokens_scheduled(tokens_pos)
            negative = clip.encode_from_tokens_scheduled(tokens_neg)
        except NotImplementedError as exc:
            hint = self._fp8_hint(exc, images_vl)
            if hint is not None:
                raise hint from exc
            raise

        if len(ref_latents) > 0:
            positive = _with_reference_latents(positive, ref_latents)
            negative = _with_reference_latents(negative, ref_latents)

        if first_wh is not None:
            latent_w, latent_h = first_wh
        elif resolution > 0:
            latent_w, latent_h = resolution, resolution
        else:
            latent_w, latent_h = 1024, 1024
        latent = {"samples": torch.zeros(
            [1, 64, latent_h // 16, latent_w // 16],
            device=_intermediate_device(),
        )}
        return (positive, negative, latent)


def _with_reference_latents(conditioning, ref_latents):
    """Append reference_latents to every conditioning entry (no node_helpers dependency)."""
    out = []
    for entry in conditioning:
        cond, data = entry[0], dict(entry[1])
        data["reference_latents"] = list(ref_latents)
        out.append([cond, data])
    return out


def _intermediate_device():
    try:
        import comfy.model_management as _mm
        return _mm.intermediate_device()
    except Exception:
        return torch.device("cpu")


class Qwen21SystemPrompt:
    """Generic text node preloaded with the instruct/edit-style system prompt. Wire its
    output into TextEncodeQwen21's `system_prompt` input to make the prompt fuse with the
    reference image (experimental / out-of-distribution). Edit the text freely."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": QWEN21_INSTRUCT_SYSTEM,
                    "tooltip": "System instruction for Qwen Image 2.1's VLM. Defaults to an instruct/edit-style "
                               "framing that fuses your prompt with the reference image. Edit as needed; "
                               "paste the plain instruction to fall back to default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = "model/conditioning/qwen image"
    DESCRIPTION = ("Text node preloaded with an instruct-style system prompt for Text Encode (Qwen Image 2.1). "
                   "Wire its output into the encoder's system_prompt input.")

    def run(self, text):
        return (text,)


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwen21": TextEncodeQwen21,
    "Qwen21SystemPrompt": Qwen21SystemPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwen21": "Text Encode (Qwen Image 2.1)",
    "Qwen21SystemPrompt": "Qwen 2.1 System Prompt",
}
