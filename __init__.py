"""ComfyUI-QwenTextEncoder — vision-aware text conditioning for Qwen Image 2.1."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serves web/ as a frontend extension (auto-growing image inputs).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
