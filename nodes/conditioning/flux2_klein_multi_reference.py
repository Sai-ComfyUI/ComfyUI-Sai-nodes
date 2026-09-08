"""Multi-reference latent conditioning for compatible edit models."""

from __future__ import annotations

import math

import comfy.utils
import node_helpers
import torch
from comfy_api.latest import io


_IMAGE_INPUT_NAMES = [f"image_{index}" for index in range(1, 17)]


def scale_image_to_total_pixels(
    image: torch.Tensor,
    upscale_method: str,
    megapixels: float,
    resolution_steps: int,
) -> torch.Tensor:
    """Scale an IMAGE tensor while preserving its aspect ratio.

    This follows ComfyUI's ``Scale Image to Total Pixels`` calculation. The
    minimum-size guard prevents an extreme aspect ratio from rounding a side to
    zero pixels.
    """

    if image.ndim != 4:
        raise ValueError(
            f"Expected IMAGE with shape [B, H, W, C], received {tuple(image.shape)}."
        )
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("Reference images must have non-zero width and height.")

    samples = image.movedim(-1, 1)
    total_pixels = megapixels * 1024 * 1024
    scale_by = math.sqrt(total_pixels / (samples.shape[3] * samples.shape[2]))
    width = max(
        resolution_steps,
        round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps,
    )
    height = max(
        resolution_steps,
        round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps,
    )

    scaled = comfy.utils.common_upscale(
        samples,
        int(width),
        int(height),
        upscale_method,
        "disabled",
    )
    return scaled.movedim(1, -1)


class Flux2KleinMultiReferenceLatent(io.ComfyNode):
    """Scale and encode multiple reference images for compatible edit models."""

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2KleinMultiReferenceLatent_Sai",
            display_name="Multi Reference Latent Ψ",
            category="Sai/Conditioning/Edit Model",
            description=(
                "Scales each reference image to the selected total pixel count, "
                "VAE-encodes it, and appends the ordered latent list to both "
                "positive and negative conditioning using ComfyUI's standard "
                "reference_latents metadata. Model support and result quality "
                "depend on the edit model. The first scaled image and the input "
                "VAE are also returned for downstream workflow use."
            ),
            search_aliases=[
                "Flux 2 Klein reference latent",
                "multi reference latent",
                "Klein image edit",
                "edit model reference latent",
            ],
            inputs=[
                io.Conditioning.Input(
                    "positive_conditioning",
                    tooltip="Positive text conditioning to receive the reference latents.",
                ),
                io.Conditioning.Input(
                    "negative_conditioning",
                    tooltip="Negative text conditioning to receive the same reference latents.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="Edit-model-compatible VAE used to encode every image.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=cls.upscale_methods,
                    default="nearest-exact",
                    tooltip="Resize filter used for every reference image.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.01,
                    max=16.0,
                    step=0.01,
                    tooltip="Target total pixels; 1.0 is 1024 × 1024 pixels.",
                ),
                io.Int.Input(
                    "resolution_steps",
                    default=1,
                    min=1,
                    max=256,
                    advanced=True,
                    tooltip=(
                        "Round each scaled dimension to this multiple. The default "
                        "matches ComfyUI's Scale Image to Total Pixels behavior."
                    ),
                ),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=_IMAGE_INPUT_NAMES,
                        min=1,
                    ),
                    tooltip=(
                        "Ordered reference images. Connecting the last visible "
                        "socket automatically adds another, up to 16 images."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_conditioning"),
                io.Conditioning.Output(display_name="negative_conditioning"),
                io.Image.Output(display_name="first_image_scaled"),
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive_conditioning,
        negative_conditioning,
        vae,
        upscale_method,
        megapixels,
        resolution_steps,
        images: io.Autogrow.Type,
    ) -> io.NodeOutput:
        images = images or {}
        reference_latents: list[torch.Tensor] = []
        first_image_scaled: torch.Tensor | None = None

        for input_name in _IMAGE_INPUT_NAMES:
            image = images.get(input_name)
            if image is None:
                continue

            scaled = scale_image_to_total_pixels(
                image,
                upscale_method,
                megapixels,
                resolution_steps,
            )
            if scaled.shape[-1] < 3:
                raise ValueError(
                    f"{input_name} must contain at least 3 color channels; "
                    f"received {scaled.shape[-1]}."
                )

            rgb = scaled[..., :3]
            if first_image_scaled is None:
                first_image_scaled = rgb
            reference_latents.append(vae.encode(rgb))

        if not reference_latents or first_image_scaled is None:
            raise ValueError("Connect at least one reference image.")

        values = {"reference_latents": reference_latents}
        positive = node_helpers.conditioning_set_values(
            positive_conditioning,
            values,
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative_conditioning,
            values,
            append=True,
        )
        return io.NodeOutput(positive, negative, first_image_scaled, vae)


__all__ = [
    "Flux2KleinMultiReferenceLatent",
    "scale_image_to_total_pixels",
]
