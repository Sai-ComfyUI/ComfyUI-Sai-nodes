"""ComfyUI integration for the LUA FLUX latent upscale adapter.

The model architecture is vendored from https://github.com/vaskers5/LUA and
remains available under its Apache-2.0 license; see ``LUA_LICENSE``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .swinir_arch import SwinIRMultiHead


HF_REPO_ID = "vaskers5/LUA-FLUX"
HF_FILENAME = "lua_flux.pth"
DOWNLOAD_OPTION = f"[download] {HF_REPO_ID}/{HF_FILENAME}"
MODEL_CATEGORY = "latent_upscale_models"

_FLUX_MODEL_CONFIG = {
    "in_chans": 16,
    "img_size": 32,
    "window_size": 16,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 360,
    "num_heads": [12, 12, 12, 12, 12, 12],
    "mlp_ratio": 2,
    "resi_connection": "1conv",
    "primary_head": "x4",
    "head_num_feat": 256,
    "heads": [
        {"name": "x2", "scale": 2, "out_chans": 16},
        {"name": "x4", "scale": 4, "out_chans": 16, "primary": True},
    ],
}

_PRECISION_DTYPES = {
    "fp32 (recommended)": torch.float32,
    "bf16": torch.bfloat16,
}


def get_model_options() -> list[str]:
    """Return the explicit download action followed by local model files."""

    local_models = folder_paths.get_filename_list(MODEL_CATEGORY)
    return [DOWNLOAD_OPTION, *[name for name in local_models if name != HF_FILENAME]]


def resolve_model_path(model_name: str) -> str:
    """Resolve a local model or download the official checkpoint on request."""

    if model_name != DOWNLOAD_OPTION:
        return folder_paths.get_full_path_or_raise(MODEL_CATEGORY, model_name)

    model_directories = folder_paths.get_folder_paths(MODEL_CATEGORY)
    if not model_directories:
        raise RuntimeError(
            "ComfyUI did not provide a latent_upscale_models directory."
        )

    target_directory = Path(model_directories[0])
    target_path = target_directory / HF_FILENAME
    if target_path.is_file():
        return str(target_path)

    target_directory.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic download. Install it or "
            f"place {HF_FILENAME} in {target_directory}."
        ) from exc

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(target_directory),
    )


def extract_lua_state_dict(
    checkpoint: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Extract the released LUA state dict from its checkpoint wrapper."""

    for key in ("params_ema", "params"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return dict(value)

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return dict(checkpoint)  # type: ignore[arg-type]

    raise ValueError(
        "Unsupported LUA checkpoint: expected 'params_ema', 'params', or a "
        "plain tensor state dict."
    )


def build_lua_model(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> comfy.model_patcher.CoreModelPatcher:
    """Build the official model without allocating random full-size weights."""

    for key, value in state_dict.items():
        if torch.is_floating_point(value) and value.dtype != dtype:
            state_dict[key] = value.to(dtype=dtype)

    with torch.device("meta"):
        model = SwinIRMultiHead(**_FLUX_MODEL_CONFIG)

    model.load_state_dict(state_dict, strict=True, assign=True)
    # ``mean`` is a plain tensor rather than a persistent buffer in upstream
    # LUA, so restore it after meta-device construction.
    model.mean = torch.zeros((1, 1, 1, 1), dtype=dtype)
    model.eval()

    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=model_management.get_torch_device(),
        offload_device=model_management.unet_offload_device(),
    )


def upscale_lua_latent(
    model: SwinIRMultiHead,
    latent: torch.Tensor,
    head: str,
) -> torch.Tensor:
    """Run the official pad, forward, and crop sequence."""

    if latent.ndim != 4:
        raise ValueError(
            "LUA-FLUX expects a 4D latent [B, 16, H, W]; "
            f"received shape {tuple(latent.shape)}."
        )
    if latent.shape[1] != 16:
        raise ValueError(
            "LUA-FLUX only supports FLUX.1 16-channel VAE latents; "
            f"received {latent.shape[1]} channels. FLUX.2 uses a different "
            "latent format and is not compatible."
        )
    if head not in model.head_scales:
        raise ValueError(
            f"Unknown LUA head {head!r}; available heads: "
            f"{list(model.head_scales)}."
        )

    window_size = int(model.window_size)
    height, width = latent.shape[-2:]
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if (pad_height and pad_height >= height) or (pad_width and pad_width >= width):
        raise ValueError(
            "LUA-FLUX reflect padding requires latent height and width large "
            f"enough for a {window_size}-pixel window; received {height}x{width}."
        )

    padded = latent
    if pad_height or pad_width:
        padded = F.pad(
            padded,
            (0, pad_width, 0, pad_height),
            mode="reflect",
        )

    scale = int(model.head_scales[head])
    output = model.forward_single_head(padded, head)
    output_height, output_width = output.shape[-2:]
    return output[
        ...,
        : output_height - pad_height * scale,
        : output_width - pad_width * scale,
    ]


class LoadLuaFluxModel(io.ComfyNode):
    """Load the official LUA-FLUX model under ComfyUI memory management."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadLuaFluxModel_Sai",
            display_name="Load LUA FLUX Model Ψ",
            category="Sai/Model/Loaders",
            description=(
                "Loads the official vaskers5/LUA-FLUX latent upscaler. Choose "
                "the download entry to fetch lua_flux.pth into ComfyUI's "
                "models/latent_upscale_models folder, or select a local file."
            ),
            search_aliases=[
                "LUA FLUX loader",
                "latent upscale adapter",
                "load latent upscaler",
            ],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=get_model_options(),
                    default=DOWNLOAD_OPTION,
                    tooltip=(
                        "Official download action or a checkpoint from "
                        "models/latent_upscale_models."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=list(_PRECISION_DTYPES),
                    default="fp32 (recommended)",
                    advanced=True,
                    tooltip=(
                        "The official implementation recommends fp32. bf16 "
                        "uses less memory but may change output quality."
                    ),
                ),
            ],
            outputs=[
                io.LatentUpscaleModel.Output(display_name="lua_model"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, precision: str) -> io.NodeOutput:
        model_path = resolve_model_path(model_name)
        checkpoint = comfy.utils.load_torch_file(model_path, safe_load=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("LUA checkpoint did not contain a state dictionary.")

        state_dict = extract_lua_state_dict(checkpoint)
        try:
            dtype = _PRECISION_DTYPES[precision]
        except KeyError as exc:
            raise ValueError(f"Unsupported precision: {precision!r}.") from exc

        return io.NodeOutput(build_lua_model(state_dict, dtype))


class LuaFluxLatentUpscale(io.ComfyNode):
    """Upscale a decoded-space FLUX.1 latent with LUA."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LuaFluxLatentUpscale_Sai",
            display_name="LUA FLUX Latent Upscale Ψ",
            category="Sai/Latent/Upscale",
            description=(
                "Upscales a standard ComfyUI FLUX.1 16-channel latent with the "
                "official LUA x2 or x4 head. Connect the result directly to VAE "
                "Decode. FLUX.2 128-channel latents are not compatible."
            ),
            search_aliases=[
                "LUA FLUX",
                "latent upscale",
                "FLUX 2K 4K",
                "latent super resolution",
            ],
            inputs=[
                io.LatentUpscaleModel.Input(
                    "lua_model",
                    tooltip="Model from Load LUA FLUX Model Ψ.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="A sampled FLUX.1 latent with shape [B, 16, H, W].",
                ),
                io.Combo.Input(
                    "scale",
                    options=["x2", "x4"],
                    default="x2",
                    tooltip="x2 targets 2K and x4 targets 4K from a 1024 base.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent_upscaled"),
            ],
        )

    @classmethod
    def execute(cls, lua_model, latent: dict, scale: str) -> io.NodeOutput:
        samples = latent.get("samples")
        if not torch.is_tensor(samples):
            raise ValueError("LATENT input must contain a tensor under 'samples'.")

        model = lua_model.model
        if not isinstance(model, SwinIRMultiHead):
            raise TypeError(
                "lua_model must come from Load LUA FLUX Model Ψ."
            )

        upscale_factor = int(model.head_scales.get(scale, 1))
        element_size = torch.tensor([], dtype=lua_model.model_dtype()).element_size()
        activation_memory = int(
            math.prod(samples.shape)
            * (upscale_factor**2)
            * element_size
            * 128
        )
        model_management.load_models_gpu(
            [lua_model],
            memory_required=activation_memory,
            force_full_load=True,
        )

        input_dtype = samples.dtype
        model_dtype = next(model.parameters()).dtype
        prepared = samples.to(
            device=lua_model.load_device,
            dtype=model_dtype,
        )
        with torch.inference_mode():
            upscaled = upscale_lua_latent(model, prepared, scale)

        result = latent.copy()
        result["samples"] = upscaled.to(
            device=model_management.intermediate_device(),
            dtype=input_dtype,
        )
        # A source-resolution noise mask is invalid after spatial upscaling.
        result.pop("noise_mask", None)
        return io.NodeOutput(result)


__all__ = [
    "DOWNLOAD_OPTION",
    "LoadLuaFluxModel",
    "LuaFluxLatentUpscale",
    "build_lua_model",
    "extract_lua_state_dict",
    "get_model_options",
    "resolve_model_path",
    "upscale_lua_latent",
]
