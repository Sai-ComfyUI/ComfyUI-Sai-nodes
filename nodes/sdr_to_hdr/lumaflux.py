"""ComfyUI V3 nodes for the native LumaFlux image pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import struct
import zlib
from typing import Mapping

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui

from ...vendor.lumaflux_native import (
    LumaFluxAdapter,
    hdr_to_sdr_bt2446c,
    hdr_to_sdr_whitepoint,
    pq_eotf_nits,
    remap_checkpoint_keys,
)


MODEL_CATEGORY = "lumaflux"
MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / MODEL_CATEGORY
folder_paths.add_model_folder_path(
    MODEL_CATEGORY,
    str(Path(folder_paths.models_dir) / MODEL_CATEGORY),
)
folder_paths.add_model_folder_path(MODEL_CATEGORY, str(MODEL_DIRECTORY))

LumaFluxModelType = io.Custom("LUMAFLUX_MODEL")
HDRImageType = io.Custom("HDR_IMAGE")
HDR_FORMAT_EXR = "OpenEXR (32-bit float, linear BT.2020)"
HDR_FORMAT_PNG = "PNG (16-bit, PQ BT.2020)"
RESIZE_MODES = ["none", "crop", "resize"]
PREVIEW_METHODS = ["BT.2446C", "Reinhard (BT.2408 white)", "Clip (BT.2408 white)"]
CICP_PQ_BT2020 = bytes([9, 16, 0, 1])


@dataclass(frozen=True)
class LumaFluxModelHandle:
    model: object
    vae: object
    clip_vision: object
    adapter: LumaFluxAdapter
    adapter_patcher: comfy.model_patcher.CoreModelPatcher
    adapter_name: str
    peak_nits: float = 1000.0


def get_adapter_options() -> list[str]:
    return [name for name in folder_paths.get_filename_list(MODEL_CATEGORY) if name.endswith(".safetensors")]


def _adapter_dtype(model) -> torch.dtype:
    dtype = model.model_dtype()
    return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16


def _validate_flux_model(model) -> object:
    try:
        diffusion = model.model.diffusion_model
        params = diffusion.params
    except AttributeError as exc:
        raise TypeError("MODEL must be a native ComfyUI FLUX.1 diffusion model.") from exc
    actual = (
        params.hidden_size,
        params.context_in_dim,
        params.vec_in_dim,
        len(diffusion.double_blocks),
        len(diffusion.single_blocks),
    )
    expected = (3072, 4096, 768, 19, 38)
    if actual != expected:
        raise ValueError(f"LumaFlux requires FLUX.1-dev architecture {expected}; received {actual}.")
    if not params.guidance_embed:
        raise ValueError("LumaFlux requires the guidance-embedded FLUX.1-dev model, not Schnell.")
    return diffusion


def load_adapter(path: str, dtype: torch.dtype) -> LumaFluxAdapter:
    checkpoint = comfy.utils.load_torch_file(path, safe_load=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("LumaFlux checkpoint must contain a tensor state dictionary.")
    state = remap_checkpoint_keys(dict(checkpoint))
    with torch.device("meta"):
        adapter = LumaFluxAdapter()
    expected = set(dict(adapter.named_parameters()))
    provided = set(state)
    missing, unexpected = sorted(expected - provided), sorted(provided - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {len(missing)} keys (first: {missing[:3]})")
        if unexpected:
            details.append(f"unexpected {len(unexpected)} keys (first: {unexpected[:3]})")
        raise ValueError("Incompatible LumaFlux adapter: " + "; ".join(details))
    for key, value in state.items():
        if torch.is_floating_point(value):
            state[key] = value.to(dtype=dtype)
    adapter.load_state_dict(state, strict=False, assign=True)
    adapter.rqs.peak_pq = torch.tensor(0.7518270962, dtype=dtype)
    adapter.eval()
    return adapter


def _double_callback(adapter, base, index, args, _extra):
    return adapter.double_block(base, index, args)


def _single_callback(adapter, base, index, args, _extra):
    return adapter.single_block(base, index, args)


def assemble_lumaflux(model, vae, clip_vision, adapter_name: str) -> LumaFluxModelHandle:
    diffusion = _validate_flux_model(model)
    if getattr(vae, "latent_channels", None) != 16 or getattr(vae, "downscale_ratio", None) != 8:
        raise ValueError("LumaFlux requires the 16-channel, 8x FLUX.1 autoencoder.")
    if getattr(clip_vision, "model_type", None) != "siglip_vision_model":
        raise ValueError("CLIP_VISION must be SigLIP SO400M patch14-384.")
    if getattr(clip_vision, "image_size", None) != 384:
        raise ValueError("LumaFlux requires a 384-pixel SigLIP vision encoder.")

    path = folder_paths.get_full_path_or_raise(MODEL_CATEGORY, adapter_name)
    adapter = load_adapter(path, _adapter_dtype(model))
    adapter_patcher = comfy.model_patcher.CoreModelPatcher(
        adapter,
        load_device=model_management.get_torch_device(),
        offload_device=model_management.unet_offload_device(),
    )
    patched = model.clone()
    for index, block in enumerate(diffusion.double_blocks):
        callback = partial(_double_callback, adapter, block, index)
        patched.set_model_patch_replace(callback, "dit", "double_block", index)
    for index, block in enumerate(diffusion.single_blocks):
        callback = partial(_single_callback, adapter, block, index)
        patched.set_model_patch_replace(callback, "dit", "single_block", index)
    return LumaFluxModelHandle(patched, vae, clip_vision, adapter, adapter_patcher, adapter_name)


def _siglip_tokens(clip_vision, image: torch.Tensor) -> torch.Tensor:
    model_management.load_model_gpu(clip_vision.patcher)
    device = clip_vision.load_device
    pixels = F.interpolate(
        image.movedim(-1, 1).to(device),
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
    )
    pixels = (pixels - 0.5) / 0.5
    dtype = next(clip_vision.model.parameters()).dtype
    output = clip_vision.model(pixel_values=pixels.to(dtype=dtype), intermediate_output=-2)
    return output[0].to(model_management.intermediate_device())


def time_shift(timesteps: torch.Tensor, factor: float) -> torch.Tensor:
    return factor * timesteps / (1.0 + (factor - 1.0) * timesteps)


def nearest_multiple(value: int, multiple: int = 16) -> int:
    """Return the nearest positive multiple, choosing upward on an exact tie."""
    if value < 1 or multiple < 1:
        raise ValueError("value and multiple must be positive")
    return max(multiple, ((value + multiple // 2) // multiple) * multiple)


def normalize_boolean_input(value, name: str = "value") -> bool:
    """Normalize native/legacy ComfyUI boolean values without string truthiness bugs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "enabled", "temporal consistency on"}:
            return True
        if normalized in {"false", "0", "no", "off", "disabled", "independent frames"}:
            return False
    raise ValueError(f"{name} must be a boolean value; received {value!r}.")


def next_multiple(value: int, multiple: int = 16) -> int:
    """Return the smallest multiple greater than or equal to a positive value."""
    if value < 1 or multiple < 1:
        raise ValueError("value and multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def resolve_lumaflux_geometry(
    width: int,
    height: int,
    resize_mode: str,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    """Return model size, output size, and a user-facing processing notice."""
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}; received {resize_mode!r}.")
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")
    if width % 16 == 0 and height % 16 == 0:
        size = (width, height)
        return size, size, f"Size unchanged: {width}×{height} (already divisible by 16)."
    if resize_mode == "none":
        raise ValueError(
            f"LumaFlux input width and height must be divisible by 16; received {width}x{height}. "
            "Select crop to edge-pad and restore the original size, or resize for nearest-×16 resampling."
        )
    if resize_mode == "crop":
        model_size = (next_multiple(width), next_multiple(height))
        return (
            model_size,
            (width, height),
            f"Processed: edge-padded {width}×{height} → {model_size[0]}×{model_size[1]}, "
            f"then cropped back to {width}×{height}.",
        )
    model_size = (nearest_multiple(width), nearest_multiple(height))
    return (
        model_size,
        model_size,
        f"Processed: resized {width}×{height} → {model_size[0]}×{model_size[1]} "
        "(nearest multiples of 16).",
    )


def prepare_lumaflux_image(
    image: torch.Tensor,
    resize_mode: str | bool,
) -> tuple[torch.Tensor, tuple[int, int], str]:
    """Apply the selected geometry policy and return BHWC input plus output crop size."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"IMAGE must be BHWC RGB; received {tuple(image.shape)}.")
    # Preserve the old helper's Python-call compatibility while the node schema
    # migrates from a boolean widget to the shared mode dropdown.
    if isinstance(resize_mode, bool):
        resize_mode = "resize" if resize_mode else "none"
    height, width = image.shape[1:3]
    model_size, output_size, notice = resolve_lumaflux_geometry(width, height, resize_mode)
    target_width, target_height = model_size
    if model_size == (width, height):
        return image, output_size, notice
    channels_first = image.movedim(-1, 1)
    if resize_mode == "crop":
        prepared = F.pad(
            channels_first,
            (0, target_width - width, 0, target_height - height),
            mode="replicate",
        )
    else:
        prepared = F.interpolate(
            channels_first,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
    return prepared.movedim(1, -1), output_size, notice


@torch.inference_mode()
def convert_sdr_to_hdr(
    handle: LumaFluxModelHandle,
    image: torch.Tensor,
    steps: int,
    seed: int,
    time_shift_factor: float,
    bridge_noise: float,
    tone_strength: float,
    resize_mode: str = "crop",
    noise: torch.Tensor | None = None,
    previous_spline_params: Mapping[str, torch.Tensor] | None = None,
    spline_ema: float = 0.0,
    return_continuity_state: bool = False,
    progress_bar: comfy.utils.ProgressBar | None = None,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0.0 <= spline_ema < 1.0:
        raise ValueError("spline_ema must be in [0, 1)")

    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"IMAGE must be BHWC RGB; received {tuple(image.shape)}.")
    source_height, source_width = image.shape[1:3]
    image, output_size, resize_notice = prepare_lumaflux_image(image, resize_mode)
    height, width = image.shape[1:3]
    image = image.clamp(0, 1)
    raw_latent = handle.vae.encode(image)
    siglip = _siglip_tokens(handle.clip_vision, image)
    model_management.load_models_gpu([handle.model, handle.adapter_patcher])
    device = handle.model.load_device
    dtype = _adapter_dtype(handle.model)
    sdr = image.movedim(-1, 1).to(device=device, dtype=dtype)
    latent = handle.model.model.process_latent_in(raw_latent.to(device=device, dtype=dtype))
    siglip = siglip.to(device=device, dtype=dtype)
    grid = (latent.shape[-2] // 2, latent.shape[-1] // 2)
    condition = handle.adapter.prepare_condition(sdr, siglip, grid)
    handle.adapter.condition = condition

    if noise is None:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
    else:
        if tuple(noise.shape) != tuple(latent.shape):
            raise ValueError(
                f"Shared bridge noise shape {tuple(noise.shape)} does not match latent shape {tuple(latent.shape)}."
            )
        noise = noise.to(device=device, dtype=latent.dtype)
    latent = latent + bridge_noise * noise
    schedule = time_shift(torch.linspace(1, 0, steps + 1, device=device, dtype=dtype), time_shift_factor)
    progress = progress_bar or comfy.utils.ProgressBar(steps)
    transformer_options = handle.model.model_options.get("transformer_options", {})
    diffusion = handle.model.model.diffusion_model
    try:
        for index in range(steps):
            time = schedule[index].expand(latent.shape[0])
            condition.time = time
            guidance = torch.ones_like(time)
            velocity = diffusion(
                latent,
                # ComfyUI's native FLUX timestep embedding performs the
                # 0..1 -> 0..1000 scaling internally. Passing time * 1000 here
                # scales it twice and produces severe packed-token artifacts.
                time,
                condition.context,
                y=condition.pooled,
                guidance=guidance,
                transformer_options=transformer_options,
            )
            latent = latent + (schedule[index + 1] - schedule[index]) * velocity
            progress.update(1)
    finally:
        handle.adapter.condition = None

    raw_result = handle.model.model.process_latent_out(latent).to(model_management.intermediate_device())
    decoded = handle.vae.decode(raw_result).movedim(-1, 1)
    model_management.load_model_gpu(handle.adapter_patcher)
    current_params = handle.adapter.rqs.spline_params(latent)
    applied_params = current_params
    if previous_spline_params is not None and spline_ema > 0.0:
        names = ("widths", "heights", "derivs")
        previous = tuple(
            previous_spline_params[name].to(device=value.device, dtype=value.dtype)
            for name, value in zip(names, current_params)
        )
        applied_params = tuple(
            spline_ema * old + (1.0 - spline_ema) * new
            for old, new in zip(previous, current_params)
        )
    hdr = handle.adapter.rqs(
        decoded.to(device=handle.adapter_patcher.load_device, dtype=dtype),
        latent.to(device=handle.adapter_patcher.load_device, dtype=dtype),
        tone_strength,
        spline_params=applied_params,
    )
    output_width, output_height = output_size
    hdr_images = hdr.movedim(1, -1)[:, :output_height, :output_width]
    result = {
        "images": hdr_images.to(model_management.intermediate_device(), dtype=torch.float32),
        "transfer": "pq",
        "primaries": "bt2020",
        "matrix": "rgb",
        "range": "full",
        "peak_nits": handle.peak_nits,
        "mastering_display": "G(0.1700,0.7970)B(0.1310,0.0460)R(0.7080,0.2920)WP(0.3127,0.3290)",
        "source_size": (source_width, source_height),
        "processed_size": output_size,
        "model_size": (width, height),
        "resize_mode": resize_mode,
        "geometry_processed": (source_width, source_height) != (width, height),
        "resize_notice": resize_notice,
    }
    if return_continuity_state:
        result["bridge_noise"] = noise.detach().cpu()
        result["spline_params"] = {
            name: value.detach().cpu()
            for name, value in zip(("widths", "heights", "derivs"), applied_params)
        }
    return result


def validate_hdr_image(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("images")):
        raise TypeError("HDR_IMAGE must contain a tensor under 'images'.")
    if payload.get("transfer") != "pq" or payload.get("primaries") != "bt2020":
        raise ValueError("HDR_IMAGE must be PQ with BT.2020 primaries.")
    return payload


def hdr_pq_to_linear_relative(images: torch.Tensor, peak_nits: float) -> torch.Tensor:
    """PQ/BT.2020 -> linear BT.2020 where 1.0 equals the mastering peak."""
    if not 0.0 < peak_nits <= 10_000.0:
        raise ValueError("peak_nits must be in (0, 10000]")
    return pq_eotf_nits(images.clamp(0.0, 1.0)) / peak_nits


def _write_hdr_sidecar(path: str, payload: dict[str, object], encoding: str) -> None:
    metadata = {
        "encoding": encoding,
        "primaries": "BT.2020",
        "white_point": "D65",
        "mastering_peak_nits": float(payload.get("peak_nits", 1000.0)),
        "mastering_display": payload.get("mastering_display"),
    }
    with open(f"{path}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def _save_exr_float32(image: torch.Tensor, path: str, peak_nits: float) -> None:
    """Write an uncompressed scanline OpenEXR with embedded HDR color data."""
    import numpy as np

    linear = hdr_pq_to_linear_relative(image, peak_nits).detach().cpu().numpy().astype("<f4")
    height, width, channels = linear.shape
    if channels != 3:
        raise ValueError(f"OpenEXR expects HWC RGB; received {tuple(linear.shape)}.")

    def cstring(value: str) -> bytes:
        return value.encode("ascii") + b"\0"

    def attribute(name: str, kind: str, value: bytes) -> bytes:
        return cstring(name) + cstring(kind) + struct.pack("<I", len(value)) + value

    channel_list = bytearray()
    for name in ("B", "G", "R"):
        channel_list += cstring(name)
        channel_list += struct.pack("<iB3xii", 2, 1, 1, 1)  # FLOAT, perceptually linear, 1x1
    channel_list += b"\0"
    bounds = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    chromaticities = struct.pack(
        "<8f",
        0.7080, 0.2920,
        0.1700, 0.7970,
        0.1310, 0.0460,
        0.3127, 0.3290,
    )
    header = bytearray()
    header += attribute("channels", "chlist", bytes(channel_list))
    header += attribute("compression", "compression", b"\0")
    header += attribute("dataWindow", "box2i", bounds)
    header += attribute("displayWindow", "box2i", bounds)
    header += attribute("lineOrder", "lineOrder", b"\0")
    header += attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
    header += attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += attribute("chromaticities", "chromaticities", chromaticities)
    header += attribute("adoptedNeutral", "v2f", struct.pack("<2f", 0.3127, 0.3290))
    header += attribute("whiteLuminance", "float", struct.pack("<f", peak_nits))
    header += attribute(
        "comments",
        "string",
        f"Linear BT.2020 RGB; 1.0 = {peak_nits:g} cd/m2; converted from LumaFlux PQ".encode("ascii"),
    )
    header += b"\0"

    prefix = struct.pack("<II", 20000630, 2) + bytes(header)
    row_payload_size = width * 3 * 4
    chunk_size = 8 + row_payload_size
    first_chunk = len(prefix) + height * 8
    offsets = b"".join(struct.pack("<Q", first_chunk + y * chunk_size) for y in range(height))
    with open(path, "wb") as exr:
        exr.write(prefix)
        exr.write(offsets)
        for y in range(height):
            # Scanline channel payload follows the header's B, G, R order.
            payload = np.ascontiguousarray(linear[y].T[[2, 1, 0]]).astype("<f4", copy=False).tobytes()
            exr.write(struct.pack("<iI", y, len(payload)))
            exr.write(payload)


def _save_png16_pq(image: torch.Tensor, path: str) -> None:
    """Write lossless 16-bit PQ RGB with a PNG cICP PQ/BT.2020 declaration."""
    import numpy as np

    rgb = (image.detach().cpu().float().clamp(0.0, 1.0).numpy() * 65535.0).round().astype(np.uint16)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"PNG expects HWC RGB; received {tuple(rgb.shape)}.")

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    rows = np.ascontiguousarray(rgb.astype(">u2")).view(np.uint8).reshape(height, width * 6)
    filtered = np.concatenate([np.zeros((height, 1), np.uint8), rows], axis=1).tobytes()
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    png = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", ihdr), chunk(b"cICP", CICP_PQ_BT2020)]
    png.extend([chunk(b"IDAT", zlib.compress(filtered, 6)), chunk(b"IEND", b"")])
    with open(path, "wb") as output:
        output.write(b"".join(png))


class ApplyLumaFluxAdapter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ApplyLumaFluxAdapter_Sai",
            display_name="Apply LumaFlux Adapter Ψ",
            category="Sai/SDR to HDR/LumaFlux",
            description="Combines native ComfyUI FLUX.1-dev, Flux AE, SigLIP Vision, and a LumaFlux adapter.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.ClipVision.Input("clip_vision"),
                io.Combo.Input("adapter_name", options=get_adapter_options()),
            ],
            outputs=[LumaFluxModelType.Output("lumaflux_model")],
        )

    @classmethod
    def execute(cls, model, vae, clip_vision, adapter_name: str) -> io.NodeOutput:
        return io.NodeOutput(assemble_lumaflux(model, vae, clip_vision, adapter_name))


class LumaFluxSdrToHdr(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LumaFluxSdrToHdr_Sai",
            display_name="LumaFlux SDR to HDR Ψ",
            category="Sai/SDR to HDR/LumaFlux",
            description="Converts BT.709 SDR images to 1000-nit PQ/BT.2020 HDR using the full LumaFlux transport.",
            inputs=[
                LumaFluxModelType.Input("lumaflux_model"),
                io.Image.Input("image"),
                io.Combo.Input(
                    "resize_mode",
                    options=RESIZE_MODES,
                    default="crop",
                    tooltip=(
                        "none: require exact ×16; crop: edge-pad for inference then restore the original size; "
                        "resize: resample to the nearest ×16 size."
                    ),
                ),
                io.Int.Input("steps", default=8, min=1, max=50),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input(
                    "shared_noise",
                    default=True,
                    tooltip=(
                        "For an IMAGE batch, reuse bridge noise and carry the previous applied RQS spline "
                        "through EMA. Disable to process every image independently."
                    ),
                ),
                io.Float.Input("spline_ema", default=0.8, min=0.0, max=0.99, step=0.01, advanced=True),
                io.Float.Input("time_shift_factor", default=1.0, min=0.01, max=10.0, step=0.01, advanced=True),
                io.Float.Input("bridge_noise", default=0.05, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("tone_strength", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True),
            ],
            outputs=[HDRImageType.Output("hdr_image")],
        )

    @classmethod
    def execute(
        cls,
        lumaflux_model,
        image,
        resize_mode,
        steps,
        seed,
        shared_noise,
        spline_ema,
        time_shift_factor,
        bridge_noise,
        tone_strength,
    ):
        if not isinstance(lumaflux_model, LumaFluxModelHandle):
            raise TypeError("lumaflux_model must come from Apply LumaFlux Adapter Ψ.")
        if not 0.0 <= spline_ema < 1.0:
            raise ValueError("spline_ema must be in [0, 1).")
        use_shared_noise = normalize_boolean_input(shared_noise, "shared_noise")
        frame_count = int(image.shape[0])
        progress = comfy.utils.ProgressBar(max(1, frame_count * steps))
        shared_bridge_noise = None
        previous_params = None
        frame_results = []
        for frame_index in range(frame_count):
            model_management.throw_exception_if_processing_interrupted()
            frame_result = convert_sdr_to_hdr(
                lumaflux_model,
                image[frame_index : frame_index + 1],
                steps,
                seed if use_shared_noise else (seed + frame_index) & 0xFFFFFFFFFFFFFFFF,
                time_shift_factor,
                bridge_noise,
                tone_strength,
                resize_mode=resize_mode,
                noise=shared_bridge_noise if use_shared_noise else None,
                previous_spline_params=previous_params if use_shared_noise else None,
                spline_ema=spline_ema if use_shared_noise else 0.0,
                return_continuity_state=True,
                progress_bar=progress,
            )
            if use_shared_noise and shared_bridge_noise is None:
                shared_bridge_noise = frame_result["bridge_noise"]
            previous_params = frame_result["spline_params"] if use_shared_noise else None
            frame_results.append(frame_result)

        result = dict(frame_results[0])
        result["images"] = torch.cat([value["images"] for value in frame_results], dim=0)
        result["temporal_consistency"] = use_shared_noise
        result.pop("bridge_noise", None)
        result.pop("spline_params", None)
        temporal_notice = (
            f"Temporal consistency: on (shared noise + applied RQS EMA {spline_ema:.2f})."
            if use_shared_noise and frame_count > 1
            else "Temporal consistency: off; frames are independent."
            if frame_count > 1
            else "Single image; temporal consistency state is not used."
        )
        return io.NodeOutput(result, ui=ui.PreviewText(f"{result['resize_notice']}\n{temporal_notice}"))


class HdrToSdrPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HdrToSdrPreviewBt2446c_Sai",
            display_name="HDR to SDR Preview Ψ",
            category="Sai/SDR to HDR/Utility",
            description="Creates a viewable BT.709 SDR preview. It does not replace or modify the HDR payload.",
            inputs=[
                HDRImageType.Input("hdr_image"),
                io.Combo.Input("method", options=PREVIEW_METHODS, default="BT.2446C"),
                io.Float.Input(
                    "white_nits",
                    default=203.0,
                    min=50.0,
                    max=1000.0,
                    step=1.0,
                    tooltip="Used by the BT.2408-white Reinhard and clip preview methods.",
                ),
            ],
            outputs=[io.Image.Output("preview")],
        )

    @classmethod
    def execute(cls, hdr_image, method: str, white_nits: float) -> io.NodeOutput:
        payload = validate_hdr_image(hdr_image)
        images = payload["images"].movedim(-1, 1)
        peak_nits = float(payload.get("peak_nits", 1000.0))
        if method == "BT.2446C":
            preview = hdr_to_sdr_bt2446c(images, peak_nits)
        elif method == "Reinhard (BT.2408 white)":
            preview = hdr_to_sdr_whitepoint(images, white_nits, peak_nits, "reinhard")
        elif method == "Clip (BT.2408 white)":
            preview = hdr_to_sdr_whitepoint(images, white_nits, peak_nits, "clip")
        else:
            raise ValueError(f"Unsupported HDR preview method: {method}")
        return io.NodeOutput(preview.movedim(1, -1))


class SaveHdrImage(io.ComfyNode):
    """Save a typed HDR payload without lossy chroma subsampling."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveHdrImage_Sai",
            display_name="Save HDR Image Ψ",
            category="Sai/Saver/HDR",
            description="Saves HDR_IMAGE as float32 linear OpenEXR or lossless 16-bit PQ PNG.",
            is_output_node=True,
            inputs=[
                HDRImageType.Input("hdr_image"),
                io.String.Input("filename_prefix", default="HDR/LumaFlux"),
                io.Combo.Input("format", options=[HDR_FORMAT_EXR, HDR_FORMAT_PNG], default=HDR_FORMAT_EXR),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, hdr_image, filename_prefix: str, format: str = HDR_FORMAT_EXR, **_deprecated) -> io.NodeOutput:
        payload = validate_hdr_image(hdr_image)
        images = payload["images"].detach().cpu().float()
        output_dir = folder_paths.get_output_directory()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )
        saved_paths = []
        for batch_index, image in enumerate(images):
            name = filename.replace("%batch_num%", str(batch_index))
            if format == HDR_FORMAT_EXR:
                output_name = f"{name}_{counter:05}.exr"
                output_path = os.path.join(folder, output_name)
                peak_nits = float(payload.get("peak_nits", 1000.0))
                _save_exr_float32(image, output_path, peak_nits)
                encoding = f"linear BT.2020 RGB; 1.0 = {peak_nits:g} cd/m2; float32"
            elif format == HDR_FORMAT_PNG:
                output_name = f"{name}_{counter:05}.png"
                output_path = os.path.join(folder, output_name)
                _save_png16_pq(image, output_path)
                encoding = "SMPTE ST 2084 (PQ) BT.2020 RGB; uint16 code values"
            else:
                raise ValueError(f"Unsupported HDR image format: {format}")
            _write_hdr_sidecar(output_path, payload, encoding)
            saved_paths.append(os.path.join(subfolder, output_name) if subfolder else output_name)
            counter += 1
        return io.NodeOutput(ui=ui.PreviewText("Saved HDR master:\n" + "\n".join(saved_paths)))


__all__ = [
    "ApplyLumaFluxAdapter",
    "CICP_PQ_BT2020",
    "HDRImageType",
    "HDR_FORMAT_EXR",
    "HDR_FORMAT_PNG",
    "HdrToSdrPreview",
    "LumaFluxModelHandle",
    "LumaFluxModelType",
    "LumaFluxSdrToHdr",
    "PREVIEW_METHODS",
    "RESIZE_MODES",
    "SaveHdrImage",
    "assemble_lumaflux",
    "convert_sdr_to_hdr",
    "get_adapter_options",
    "hdr_pq_to_linear_relative",
    "load_adapter",
    "nearest_multiple",
    "normalize_boolean_input",
    "prepare_lumaflux_image",
    "resolve_lumaflux_geometry",
    "time_shift",
    "validate_hdr_image",
]
