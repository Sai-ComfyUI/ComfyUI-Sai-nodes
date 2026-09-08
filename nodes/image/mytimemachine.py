"""ComfyUI inference nodes for MyTimeMachine facial age transformation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping
import zipfile

import comfy.utils
import comfy.model_management as model_management
import comfy.model_patcher
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from torch import nn

from ...vendor.mytimemachine.models.psp import pSp


MODEL_CATEGORY = "mytimemachine"
MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "mytimemachine"
NO_CHECKPOINT = "[place a .pt/.pth checkpoint in models/mytimemachine]"
NO_GLOBAL_CHECKPOINT = "[none - direct checkpoint inference]"
NO_PERSONALIZED_CHECKPOINT = "[none - use base model only]"
AUXILIARY_CHECKPOINT_FILENAMES = {
    "dex_age_classifier.pth",
    "model_ir_se50.pth",
    "lpips_alex_v0.1.pth",
    "psp_ffhq_encode.pt",
    "stylegan2-ffhq-config-f.pt",
}
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}
MyTimeMachineModelType = io.Custom("SAI_MYTIMEMACHINE_MODEL")

MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path(MODEL_CATEGORY, str(MODEL_DIRECTORY))


def get_checkpoint_options() -> list[str]:
    """List user-supplied checkpoints recursively from the project model folder."""

    checkpoints = [
        name
        for name in folder_paths.get_filename_list(MODEL_CATEGORY)
        if Path(name).suffix.lower() in CHECKPOINT_EXTENSIONS
        and Path(name).name.lower() not in AUXILIARY_CHECKPOINT_FILENAMES
    ]
    return checkpoints or [NO_CHECKPOINT]


def get_global_checkpoint_options() -> list[str]:
    """Compatibility alias for the former global-checkpoint selector."""

    return get_personalized_checkpoint_options()


def is_personalized_checkpoint(checkpoint: str) -> bool | None:
    """Classify checkpoints without loading their large tensor storages."""

    if Path(checkpoint).name.lower() == "sam_ffhq_aging.pt":
        return False
    checkpoint_path = folder_paths.get_full_path(MODEL_CATEGORY, checkpoint)
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            metadata_name = next(
                (
                    name
                    for name in archive.namelist()
                    if name == "data.pkl" or name.endswith("/data.pkl")
                ),
                None,
            )
            if metadata_name is not None:
                return b"blender" in archive.read(metadata_name)
    return None


def get_base_checkpoint_options() -> list[str]:
    checkpoints = get_checkpoint_options()
    base = [name for name in checkpoints if is_personalized_checkpoint(name) is not True]
    return base or checkpoints


def get_personalized_checkpoint_options() -> list[str]:
    return [NO_PERSONALIZED_CHECKPOINT, *[
        name
        for name in get_checkpoint_options()
        if name != NO_CHECKPOINT and is_personalized_checkpoint(name) is not False
    ]]


def get_default_base_checkpoint(checkpoints: list[str]) -> str:
    """Prefer the official global SAM aging model when it is available."""

    for checkpoint in checkpoints:
        if Path(checkpoint).name.lower() == "sam_ffhq_aging.pt":
            return checkpoint
    return checkpoints[0]


def resolve_checkpoint_roles(
    checkpoint: str,
    global_checkpoint: str,
) -> tuple[str, str | None]:
    """Resolve stable input IDs to the base and optional personalized roles."""

    base_checkpoint = checkpoint
    personalized_checkpoint = global_checkpoint
    if personalized_checkpoint in {
        NO_PERSONALIZED_CHECKPOINT,
        NO_GLOBAL_CHECKPOINT,
    }:
        return base_checkpoint, None

    # Compatibility for workflows saved with the former UI contract:
    # checkpoint=personalized and global_checkpoint=sam_ffhq_aging.pt.
    if (
        Path(personalized_checkpoint).name.lower() == "sam_ffhq_aging.pt"
        and Path(base_checkpoint).name.lower() != "sam_ffhq_aging.pt"
    ):
        return personalized_checkpoint, base_checkpoint
    return base_checkpoint, personalized_checkpoint


def _load_checkpoint_options(
    checkpoint_path: str,
    *,
    encoder_only: bool = False,
) -> Namespace:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("opts"), Mapping):
        raise ValueError(
            "Unsupported MyTimeMachine checkpoint: expected a mapping containing 'opts'."
        )
    options = dict(checkpoint["opts"])
    del checkpoint
    options["checkpoint_path"] = checkpoint_path
    options["device"] = "cpu"
    options["encoder_only"] = encoder_only
    return Namespace(**options)


def _build_psp(checkpoint_path: str, *, encoder_only: bool = False) -> pSp:
    model = pSp(
        _load_checkpoint_options(checkpoint_path, encoder_only=encoder_only)
    )
    model.eval()
    return model


class MyTimeMachineBundle(nn.Module):
    """Local model plus an optional encoder-only global SAM prior."""

    def __init__(self, local_model: pSp, global_model: pSp | None = None) -> None:
        super().__init__()
        self.local_model = local_model
        self.global_model = global_model

    def get_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def load_mytimemachine_bundle(
    checkpoint_path: str,
    global_checkpoint_path: str | None,
) -> MyTimeMachineBundle:
    local_model = _build_psp(checkpoint_path)
    global_model = None
    if global_checkpoint_path is not None:
        # Only the global encoder is needed to obtain the blending latent.
        global_model = _build_psp(global_checkpoint_path, encoder_only=True)
    return MyTimeMachineBundle(local_model, global_model).eval()


def prepare_conditioned_image(image: torch.Tensor, target_age: int) -> torch.Tensor:
    """Convert a ComfyUI IMAGE batch to SAM's normalized 4-channel input."""

    if image.ndim != 4:
        raise ValueError(f"IMAGE must have shape [B,H,W,C], received {tuple(image.shape)}.")
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    elif image.shape[-1] < 3:
        raise ValueError("MyTimeMachine requires an RGB or grayscale IMAGE input.")
    rgb = image[..., :3].movedim(-1, 1)
    rgb = F.interpolate(rgb, size=(256, 256), mode="bilinear", align_corners=False)
    rgb = rgb * 2.0 - 1.0
    age = torch.full(
        (rgb.shape[0], 1, 256, 256),
        float(target_age) / 100.0,
        dtype=rgb.dtype,
        device=rgb.device,
    )
    return torch.cat((rgb, age), dim=1)


def run_age_transform(
    bundle: MyTimeMachineBundle,
    conditioned: torch.Tensor,
    target_age: int,
) -> torch.Tensor:
    """Run either direct SAM inference or personalized latent blending."""

    local_latent = bundle.local_model.encode(conditioned)
    if bundle.global_model is not None:
        global_latent = bundle.global_model.encode(conditioned)
        ages = torch.full(
            (conditioned.shape[0],),
            float(target_age) / 100.0,
            dtype=conditioned.dtype,
            device=conditioned.device,
        )
        latent = bundle.local_model.blender(local_latent, global_latent, ages)
    else:
        latent = local_latent

    images, _ = bundle.local_model.decoder(
        [latent],
        input_is_latent=True,
        randomize_noise=False,
        return_latents=True,
    )
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0).movedim(1, -1)


def build_target_ages(
    target_age: int,
    age_mode: str,
    range_start_age: int,
    range_end_age: int,
    range_step: int,
) -> list[int]:
    """Resolve the node widgets to one age or an inclusive age sequence."""

    if age_mode == "single":
        return [int(target_age)]
    if age_mode != "range":
        raise ValueError(f"Unsupported age_mode: {age_mode!r}.")

    start = int(range_start_age)
    end = int(range_end_age)
    step = max(1, int(range_step))
    direction = 1 if end >= start else -1
    ages = list(range(start, end + direction, direction * step))
    if ages[-1] != end:
        ages.append(end)
    return ages


class LoadMyTimeMachineModelSai(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        checkpoints = get_base_checkpoint_options()
        personalized_checkpoints = get_personalized_checkpoint_options()
        return io.Schema(
            node_id="LoadMyTimeMachineModel_Sai",
            display_name="Load MyTM Re-ageing Model Ψ",
            category="Sai/Re-ageing/MyTimeMachine",
            description=(
                "Loads a general SAM aging base model and optionally a personalized "
                "MyTimeMachine checkpoint for identity-specific blending."
            ),
            search_aliases=["MyTimeMachine loader", "SAM age model", "facial aging"],
            inputs=[
                io.Combo.Input(
                    "checkpoint",
                    display_name="base_checkpoint",
                    options=checkpoints,
                    default=get_default_base_checkpoint(checkpoints),
                    tooltip=(
                        "General aging model. Normally use sam_ffhq_aging.pt; it is "
                        "also the global aging prior for personalized inference."
                    ),
                ),
                io.Combo.Input(
                    "global_checkpoint",
                    display_name="personalized_checkpoint",
                    options=personalized_checkpoints,
                    default=NO_PERSONALIZED_CHECKPOINT,
                    tooltip=(
                        "Optional identity-specific MyTimeMachine checkpoint, such as "
                        "the Al Pacino iteration_10000.pt example."
                    ),
                ),
            ],
            outputs=[MyTimeMachineModelType.Output(display_name="mytimemachine_model")],
        )

    @classmethod
    def execute(cls, checkpoint: str, global_checkpoint: str) -> io.NodeOutput:
        if checkpoint == NO_CHECKPOINT:
            raise FileNotFoundError(
                f"No checkpoint found. Place model files in {MODEL_DIRECTORY}."
            )
        base_checkpoint, personalized_checkpoint = resolve_checkpoint_roles(
            checkpoint,
            global_checkpoint,
        )
        if personalized_checkpoint is None:
            local_path = folder_paths.get_full_path_or_raise(
                MODEL_CATEGORY,
                base_checkpoint,
            )
            global_path = None
        else:
            local_path = folder_paths.get_full_path_or_raise(
                MODEL_CATEGORY,
                personalized_checkpoint,
            )
            global_path = folder_paths.get_full_path_or_raise(
                MODEL_CATEGORY,
                base_checkpoint,
            )
        bundle = load_mytimemachine_bundle(local_path, global_path)
        patcher = comfy.model_patcher.CoreModelPatcher(
            bundle,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        return io.NodeOutput(patcher)


class MyTimeMachineAgeTransformSai(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyTimeMachineAgeTransform_Sai",
            display_name="MyTM Re-ageing Ψ",
            category="Sai/Re-ageing/MyTimeMachine",
            description=(
                "Transforms an aligned FFHQ-style face to a target age. Output is "
                "1024x1024. Face alignment should be performed upstream."
            ),
            search_aliases=["facial age", "aging", "de-aging", "SAM", "MyTimeMachine"],
            inputs=[
                MyTimeMachineModelType.Input("mytimemachine_model"),
                io.Image.Input(
                    "image",
                    tooltip="Aligned, tightly cropped FFHQ-style face image.",
                ),
                io.Int.Input(
                    "target_age",
                    default=30,
                    min=0,
                    max=100,
                    step=1,
                ),
                io.Combo.Input(
                    "age_mode",
                    options=["single", "range"],
                    default="single",
                    tooltip=(
                        "single uses target_age; range emits an age-ordered IMAGE "
                        "batch from range_start_age through range_end_age."
                    ),
                ),
                io.Int.Input(
                    "range_start_age",
                    default=10,
                    min=0,
                    max=100,
                    step=1,
                ),
                io.Int.Input(
                    "range_end_age",
                    default=30,
                    min=0,
                    max=100,
                    step=1,
                ),
                io.Int.Input(
                    "range_step",
                    default=1,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Positive interval; direction is inferred from start and end.",
                ),
            ],
            outputs=[io.Image.Output(display_name="aged_image")],
        )

    @classmethod
    def execute(
        cls,
        mytimemachine_model: Any,
        image: torch.Tensor,
        target_age: int,
        age_mode: str = "single",
        range_start_age: int = 10,
        range_end_age: int = 30,
        range_step: int = 1,
    ) -> io.NodeOutput:
        bundle = mytimemachine_model.model
        if not isinstance(bundle, MyTimeMachineBundle):
            raise TypeError(
                "mytimemachine_model must come from Load MyTM Re-ageing Model Ψ."
            )
        dtype = mytimemachine_model.model_dtype()
        device = mytimemachine_model.load_device
        batch = int(image.shape[0]) if image.ndim == 4 else 1
        activation_memory = batch * 3 * 1024 * 1024 * 4 * 20
        model_management.load_models_gpu(
            [mytimemachine_model],
            memory_required=activation_memory,
            force_full_load=True,
        )
        target_ages = build_target_ages(
            target_age,
            age_mode,
            range_start_age,
            range_end_age,
            range_step,
        )
        progress = comfy.utils.ProgressBar(len(target_ages))
        intermediate_device = model_management.intermediate_device()
        results = []
        with torch.inference_mode():
            for index, age in enumerate(target_ages, start=1):
                model_management.throw_exception_if_processing_interrupted()
                conditioned = prepare_conditioned_image(image, age).to(
                    device=device,
                    dtype=dtype,
                )
                result = run_age_transform(bundle, conditioned, age)
                results.append(
                    result.to(device=intermediate_device, dtype=torch.float32)
                )
                progress.update_absolute(index, len(target_ages))
        return io.NodeOutput(torch.cat(results, dim=0))


__all__ = [
    "LoadMyTimeMachineModelSai",
    "MyTimeMachineAgeTransformSai",
    "MyTimeMachineBundle",
    "NO_PERSONALIZED_CHECKPOINT",
    "get_default_base_checkpoint",
    "get_base_checkpoint_options",
    "get_checkpoint_options",
    "get_personalized_checkpoint_options",
    "is_personalized_checkpoint",
    "resolve_checkpoint_roles",
    "load_mytimemachine_bundle",
    "build_target_ages",
    "prepare_conditioned_image",
    "run_age_transform",
]
