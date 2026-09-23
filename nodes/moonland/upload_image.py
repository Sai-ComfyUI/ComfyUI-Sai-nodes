"""Ensure that a ComfyUI image exists as a Moonland resource."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import mimetypes
import os
from typing import Any

import folder_paths
import numpy as np
from comfy_api.latest import io
from PIL import Image
import torch

from ...server.moonland_bridge import get_primary_connection, request_bridge_action


MoonlandResourceType = io.Custom("MOONLAND_RESOURCE")
SUPPORTED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def input_image_files() -> list[str]:
    input_directory = folder_paths.get_input_directory()
    files = [
        name
        for name in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, name))
    ]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


def resolve_workspace_id(workspace_id: str) -> str:
    resolved = workspace_id.strip()
    if not resolved:
        connection = get_primary_connection()
        resolved = connection["moonland"].get("workspace_id") or ""
    if not resolved:
        raise ValueError(
            "The connected Moonland tab did not report a workspace ID. Open the target workspace page or provide workspace_id."
        )
    return resolved


async def ensure_image_resource(
    data: bytes,
    *,
    content_type: str,
    filename: str,
    workspace_id: str,
) -> io.NodeOutput:
    result: dict[str, Any] = await request_bridge_action(
        "resource.ensureImage",
        {
            "base64": base64.b64encode(data).decode("ascii"),
            "contentType": content_type,
            "filename": filename.strip()[:200] or "comfyui-image.png",
            "workspaceId": resolve_workspace_id(workspace_id),
        },
    )
    resource_id = result.get("resourceId")
    disposition = result.get("disposition")
    sha256 = result.get("sha256")
    if not all(isinstance(value, str) and value for value in (resource_id, disposition, sha256)):
        raise RuntimeError("Moonland Bridge returned an incomplete resource result.")
    return io.NodeOutput(result, resource_id, disposition, sha256)


def image_to_png_bytes(image: torch.Tensor) -> bytes:
    """Encode one ComfyUI IMAGE item as a deterministic, metadata-free PNG."""

    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("image must be a ComfyUI IMAGE batch with shape [B,H,W,C].")
    if image.shape[0] != 1:
        raise ValueError("Moonland Ensure Image Resource currently accepts exactly one image per execution.")
    if image.shape[-1] not in {1, 3, 4}:
        raise ValueError("Moonland image input must have 1, 3, or 4 channels.")
    array = image[0].detach().to(device="cpu", dtype=torch.float32).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    output = BytesIO()
    Image.fromarray(array, mode=mode).save(output, format="PNG", optimize=False, compress_level=6)
    return output.getvalue()


class MoonlandEnsureImageResourceSai(io.ComfyNode):
    """Use the signed-in Moonland tab to reuse or upload one image resource."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MoonlandEnsureImageResource_Sai",
            display_name="Moonland Ensure Image Resource 峔",
            category="Sai/Moonland",
            is_output_node=True,
            description=(
                "Checks Moonland by normalized SHA-256 and reuses an existing image resource. "
                "Uploads through the signed-in browser tab only when Moonland reports it missing."
            ),
            search_aliases=["Moonland upload", "Moonland resource", "Moonland image"],
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "filename",
                    default="comfyui-image.png",
                    tooltip="Library filename only; it does not affect Moonland's content fingerprint.",
                ),
                io.String.Input(
                    "workspace_id",
                    default="",
                    advanced=True,
                    tooltip="Leave empty to use the workspace reported by the connected Moonland tab.",
                ),
            ],
            outputs=[
                MoonlandResourceType.Output(display_name="moonland_resource"),
                io.String.Output(display_name="resource_id"),
                io.String.Output(display_name="disposition"),
                io.String.Output(display_name="sha256"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        filename: str = "comfyui-image.png",
        workspace_id: str = "",
    ) -> io.NodeOutput:
        png = image_to_png_bytes(image)
        return await ensure_image_resource(
            png,
            content_type="image/png",
            filename=filename,
            workspace_id=workspace_id,
        )


class MoonlandEnsureImageFileSai(io.ComfyNode):
    """Ensure an original ComfyUI input file exists as a Moonland resource."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MoonlandEnsureImageFile_Sai",
            display_name="Moonland Ensure Image File 峔",
            category="Sai/Moonland",
            is_output_node=True,
            description=(
                "Sends the original PNG, JPEG, or WebP bytes from ComfyUI input to Moonland. "
                "Use this when an existing Moonland upload must be matched exactly."
            ),
            search_aliases=["Moonland file upload", "Moonland original image", "Moonland deduplicate"],
            inputs=[
                io.Combo.Input(
                    "image_file",
                    options=input_image_files(),
                    upload=io.UploadType.image,
                    image_folder=io.FolderType.input,
                    tooltip="Original file bytes are preserved until Moonland performs its own normalization.",
                ),
                io.String.Input(
                    "workspace_id",
                    default="",
                    advanced=True,
                    tooltip="Leave empty to use the workspace reported by the connected Moonland tab.",
                ),
            ],
            outputs=[
                MoonlandResourceType.Output(display_name="moonland_resource"),
                io.String.Output(display_name="resource_id"),
                io.String.Output(display_name="disposition"),
                io.String.Output(display_name="sha256"),
            ],
        )

    @classmethod
    def validate_inputs(cls, image_file: str, workspace_id: str = "") -> bool | str:
        if not folder_paths.exists_annotated_filepath(image_file):
            return f"Invalid image file: {image_file}"
        content_type = mimetypes.guess_type(image_file)[0]
        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            return "Moonland image files must be PNG, JPEG, or WebP."
        return True

    @classmethod
    def fingerprint_inputs(cls, image_file: str, workspace_id: str = "") -> str:
        path = folder_paths.get_annotated_filepath(image_file)
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{digest.hexdigest()}:{workspace_id.strip()}"

    @classmethod
    async def execute(cls, image_file: str, workspace_id: str = "") -> io.NodeOutput:
        path = folder_paths.get_annotated_filepath(image_file)
        content_type = mimetypes.guess_type(path)[0]
        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            raise ValueError("Moonland image files must be PNG, JPEG, or WebP.")
        with open(path, "rb") as source:
            data = source.read()
        return await ensure_image_resource(
            data,
            content_type=content_type,
            filename=os.path.basename(path),
            workspace_id=workspace_id,
        )


__all__ = [
    "MoonlandEnsureImageFileSai",
    "MoonlandEnsureImageResourceSai",
    "MoonlandResourceType",
    "ensure_image_resource",
    "image_to_png_bytes",
    "input_image_files",
]
