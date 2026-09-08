"""Automatic FFHQ-style face alignment and inverse compositing nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io


ALIGN_SIZE = 1024
YUNET_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "mytimemachine"
    / "face_detection_yunet_2026may.onnx"
)
FaceReframeContextType = io.Custom("SAI_FACE_REFRAME_CONTEXT")


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Automatic face alignment requires OpenCV with FaceDetectorYN support."
        ) from exc
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV FaceDetectorYN is unavailable; OpenCV >= 4.10 is required.")
    return cv2


def detect_face_landmarks(
    rgb: np.ndarray,
    confidence: float,
    face_index: int,
    max_detection_side: int = 1024,
) -> np.ndarray:
    """Return YuNet's five landmarks for the selected face in source pixels."""

    if not YUNET_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_MODEL_PATH}. See models/mytimemachine/README.md."
        )
    cv2 = _cv2()
    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_detection_side) / max(height, width))
    if scale < 1.0:
        detection_rgb = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_rgb = rgb
    det_h, det_w = detection_rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (det_w, det_h),
        float(confidence),
        0.3,
        5000,
    )
    _, faces = detector.detect(cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        raise ValueError(
            "No face was detected. Try lowering detector_confidence or use a clearer input."
        )
    ordered = sorted(faces, key=lambda row: float(row[2] * row[3]), reverse=True)
    if face_index >= len(ordered):
        raise ValueError(
            f"face_index {face_index} is unavailable; detected {len(ordered)} face(s)."
        )
    # YuNet: bbox, right eye, left eye, nose, right mouth, left mouth, score.
    landmarks = np.asarray(ordered[face_index][4:14], dtype=np.float32).reshape(5, 2)
    return landmarks / scale


def calculate_ffhq_quad(landmarks: np.ndarray, crop_scale: float) -> np.ndarray:
    """Approximate the official FFHQ oriented crop from YuNet five-point landmarks."""

    eyes = sorted(landmarks[:2], key=lambda point: float(point[0]))
    mouths = sorted(landmarks[3:5], key=lambda point: float(point[0]))
    eye_left, eye_right = np.asarray(eyes[0]), np.asarray(eyes[1])
    mouth_left, mouth_right = np.asarray(mouths[0]), np.asarray(mouths[1])
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x_axis = eye_to_eye - np.flipud(eye_to_mouth) * np.asarray([-1.0, 1.0])
    norm = float(np.hypot(*x_axis))
    if norm < 1e-6:
        raise ValueError("Detected face landmarks are degenerate and cannot be aligned.")
    x_axis /= norm
    x_axis *= max(
        float(np.hypot(*eye_to_eye)) * 2.0,
        float(np.hypot(*eye_to_mouth)) * 1.8,
    ) * float(crop_scale)
    y_axis = np.flipud(x_axis) * np.asarray([-1.0, 1.0])
    center = eye_avg + eye_to_mouth * 0.1
    return np.stack(
        [
            center - x_axis - y_axis,
            center - x_axis + y_axis,
            center + x_axis + y_axis,
            center + x_axis - y_axis,
        ]
    ).astype(np.float32)


def align_face(rgb: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    edge = float(ALIGN_SIZE - 1)
    destination = np.asarray(
        [[0.0, 0.0], [0.0, edge], [edge, edge], [edge, 0.0]],
        dtype=np.float32,
    )
    source_to_aligned = cv2.getPerspectiveTransform(quad, destination)
    aligned = cv2.warpPerspective(
        rgb,
        source_to_aligned,
        (ALIGN_SIZE, ALIGN_SIZE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, np.linalg.inv(source_to_aligned).astype(np.float32)


def make_blend_mask(mask_shape: str, mask_scale: float, feather: float) -> np.ndarray:
    """Create a soft 1024-square mask for inverse warping."""

    yy, xx = np.mgrid[0:ALIGN_SIZE, 0:ALIGN_SIZE].astype(np.float32)
    x = (xx + 0.5) / ALIGN_SIZE
    y = (yy + 0.5) / ALIGN_SIZE
    feather = max(float(feather), 1.0 / ALIGN_SIZE)
    if mask_shape == "oval (recommended)":
        radius_x = 0.47 * float(mask_scale)
        radius_y = 0.55 * float(mask_scale)
        distance = np.sqrt(((x - 0.5) / radius_x) ** 2 + ((y - 0.5) / radius_y) ** 2)
        mask = np.clip((1.0 - distance) / feather, 0.0, 1.0)
    elif mask_shape == "full crop":
        edge_distance = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
        mask = np.clip(edge_distance / feather, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported mask shape: {mask_shape!r}.")
    return mask.astype(np.float32)


def match_patch_color(
    patch: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return patch
    weights = mask[..., None].astype(np.float32)
    total = max(float(weights.sum()), 1.0)
    patch_f = patch.astype(np.float32) / 255.0
    reference_f = reference.astype(np.float32) / 255.0
    patch_mean = (patch_f * weights).sum(axis=(0, 1)) / total
    reference_mean = (reference_f * weights).sum(axis=(0, 1)) / total
    patch_var = (((patch_f - patch_mean) ** 2) * weights).sum(axis=(0, 1)) / total
    reference_var = (((reference_f - reference_mean) ** 2) * weights).sum(axis=(0, 1)) / total
    ratio = np.sqrt(reference_var + 1e-6) / np.sqrt(patch_var + 1e-6)
    ratio = np.clip(ratio, 0.67, 1.5)
    matched = (patch_f - patch_mean) * ratio + reference_mean
    mixed = patch_f * (1.0 - strength) + matched * strength
    return np.clip(np.rint(mixed * 255.0), 0, 255).astype(np.uint8)


def restore_face(
    original: np.ndarray,
    aligned_source: np.ndarray,
    processed_face: np.ndarray,
    aligned_to_source: np.ndarray,
    mask_shape: str,
    mask_scale: float,
    feather: float,
    color_match_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    source_h, source_w = original.shape[:2]
    if processed_face.shape[:2] != (ALIGN_SIZE, ALIGN_SIZE):
        processed_face = cv2.resize(
            processed_face,
            (ALIGN_SIZE, ALIGN_SIZE),
            interpolation=cv2.INTER_LANCZOS4,
        )
    mask = make_blend_mask(mask_shape, mask_scale, feather)
    processed_face = match_patch_color(
        processed_face,
        aligned_source,
        mask,
        float(color_match_strength),
    )
    restored_patch = cv2.warpPerspective(
        processed_face,
        aligned_to_source,
        (source_w, source_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32) / 255.0
    restored_mask = cv2.warpPerspective(
        mask,
        aligned_to_source,
        (source_w, source_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).clip(0.0, 1.0)
    original_f = original.astype(np.float32) / 255.0
    composite = restored_patch * restored_mask[..., None] + original_f * (
        1.0 - restored_mask[..., None]
    )
    return np.clip(composite, 0.0, 1.0), restored_mask


class MyTimeMachineFaceAlignSai(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyTimeMachineFaceAlign_Sai",
            display_name="YuNet FFHQ Face Align Ψ",
            category="Sai/Face Alignment/YuNet FFHQ",
            description=(
                "Detects a face with YuNet and creates an FFHQ-style 1024x1024 "
                "aligned crop plus context for restoring it to the source image."
            ),
            search_aliases=["face crop", "face align", "FFHQ align", "YuNet"],
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("face_index", default=0, min=0, max=32, step=1),
                io.Float.Input(
                    "crop_scale",
                    default=1.0,
                    min=0.75,
                    max=1.5,
                    step=0.01,
                    tooltip="Higher values include more context and make the head smaller.",
                ),
                io.Float.Input(
                    "detector_confidence",
                    default=0.8,
                    min=0.3,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="aligned_face"),
                FaceReframeContextType.Output(display_name="reframe_context"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        face_index: int,
        crop_scale: float,
        detector_confidence: float,
    ) -> io.NodeOutput:
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("Face Align expects an RGB IMAGE batch [B,H,W,C].")
        original = image[..., :3].detach().to(device="cpu", dtype=torch.float32)
        aligned_images: list[torch.Tensor] = []
        inverse_matrices: list[torch.Tensor] = []
        for batch_index, item in enumerate(original):
            rgb = np.clip(np.rint(item.numpy() * 255.0), 0, 255).astype(np.uint8)
            try:
                landmarks = detect_face_landmarks(
                    rgb,
                    detector_confidence,
                    face_index,
                )
            except ValueError as exc:
                raise ValueError(f"Batch item {batch_index}: {exc}") from exc
            quad = calculate_ffhq_quad(landmarks, crop_scale)
            aligned, inverse = align_face(rgb, quad)
            aligned_images.append(torch.from_numpy(aligned.astype(np.float32) / 255.0))
            inverse_matrices.append(torch.from_numpy(inverse))
        aligned_batch = torch.stack(aligned_images)
        context: dict[str, Any] = {
            "original": original,
            "aligned_source": aligned_batch,
            "aligned_to_source": torch.stack(inverse_matrices),
        }
        return io.NodeOutput(aligned_batch, context)


class MyTimeMachineFaceRestoreSai(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyTimeMachineFaceRestore_Sai",
            display_name="YuNet FFHQ Face Restore Ψ",
            category="Sai/Face Alignment/YuNet FFHQ",
            description=(
                "Inverse-warps a processed aligned face into the original image "
                "with a feathered mask and optional color matching."
            ),
            search_aliases=["paste face", "restore face crop", "face composite"],
            inputs=[
                io.Image.Input("processed_face"),
                FaceReframeContextType.Input("reframe_context"),
                io.Combo.Input(
                    "mask_shape",
                    options=["oval (recommended)", "full crop"],
                    default="oval (recommended)",
                ),
                io.Float.Input(
                    "mask_scale",
                    default=1.0,
                    min=0.7,
                    max=1.2,
                    step=0.01,
                ),
                io.Float.Input(
                    "feather",
                    default=0.12,
                    min=0.01,
                    max=0.4,
                    step=0.01,
                    tooltip="Soft-edge width as a fraction of the aligned crop.",
                ),
                io.Float.Input(
                    "color_match_strength",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="composited_image"),
                io.Mask.Output(display_name="paste_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        processed_face: torch.Tensor,
        reframe_context: dict[str, Any],
        mask_shape: str,
        mask_scale: float,
        feather: float,
        color_match_strength: float,
    ) -> io.NodeOutput:
        original = reframe_context.get("original")
        aligned_source = reframe_context.get("aligned_source")
        inverse = reframe_context.get("aligned_to_source")
        if not all(torch.is_tensor(value) for value in (original, aligned_source, inverse)):
            raise ValueError("Invalid reframe_context; connect YuNet FFHQ Face Align Ψ.")
        processed = processed_face[..., :3].detach().to(device="cpu", dtype=torch.float32)
        count = int(original.shape[0])
        if processed.shape[0] not in (1, count):
            raise ValueError(
                f"processed_face batch must be 1 or {count}; received {processed.shape[0]}."
            )
        composites: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for index in range(count):
            patch_index = 0 if processed.shape[0] == 1 else index
            original_np = np.clip(
                np.rint(original[index].numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            aligned_np = np.clip(
                np.rint(aligned_source[index].numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            processed_np = np.clip(
                np.rint(processed[patch_index].numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            composite, mask = restore_face(
                original_np,
                aligned_np,
                processed_np,
                inverse[index].numpy(),
                mask_shape,
                mask_scale,
                feather,
                color_match_strength,
            )
            composites.append(torch.from_numpy(composite))
            masks.append(torch.from_numpy(mask))
        return io.NodeOutput(torch.stack(composites), torch.stack(masks))


__all__ = [
    "MyTimeMachineFaceAlignSai",
    "MyTimeMachineFaceRestoreSai",
    "align_face",
    "calculate_ffhq_quad",
    "detect_face_landmarks",
    "make_blend_mask",
    "match_patch_color",
    "restore_face",
]
