"""Tests for the MyTimeMachine ComfyUI integration."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_sai_nodes"
if PACKAGE_NAME not in sys.modules:
    PACKAGE_SPEC = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    assert PACKAGE_SPEC is not None and PACKAGE_SPEC.loader is not None
    PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
    sys.modules[PACKAGE_NAME] = PACKAGE
    PACKAGE_SPEC.loader.exec_module(PACKAGE)

from comfyui_sai_nodes import SaiNodesExtension
from comfyui_sai_nodes.nodes.image.mytimemachine import (
    LoadMyTimeMachineModelSai,
    MODEL_DIRECTORY,
    NO_GLOBAL_CHECKPOINT,
    NO_PERSONALIZED_CHECKPOINT,
    MyTimeMachineAgeTransformSai,
    MyTimeMachineBundle,
    build_target_ages,
    get_default_base_checkpoint,
    get_base_checkpoint_options,
    get_checkpoint_options,
    prepare_conditioned_image,
    resolve_checkpoint_roles,
    run_age_transform,
)
from comfyui_sai_nodes.nodes.image.face_reframe import (
    MyTimeMachineFaceAlignSai,
    MyTimeMachineFaceRestoreSai,
)
from comfyui_sai_nodes.vendor.mytimemachine.models.stylegan2.op import (
    FusedLeakyReLU,
    upfirdn2d,
)


class FakeLocalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.blender_ages = None

    def encode(self, conditioned: torch.Tensor) -> torch.Tensor:
        return torch.zeros((conditioned.shape[0], 18, 512), device=conditioned.device)

    def blender(self, local, global_latent, ages):
        self.blender_ages = ages
        return local + global_latent + 1.0

    def decoder(self, styles, **kwargs):
        batch = styles[0].shape[0]
        image = torch.zeros((batch, 3, 8, 8), device=styles[0].device)
        return image, styles[0]


class FakeGlobalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def encode(self, conditioned: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (conditioned.shape[0], 18, 512),
            2.0,
            device=conditioned.device,
        )


class MyTimeMachineTests(unittest.TestCase):
    def test_model_drop_directory_exists(self) -> None:
        self.assertTrue(MODEL_DIRECTORY.is_dir())
        self.assertTrue((MODEL_DIRECTORY / "README.md").is_file())

    def test_loader_hides_training_and_auxiliary_models(self) -> None:
        files = [
            "README.md",
            "sam_ffhq_aging.pt",
            "personalized\\iteration_10000.pt",
            "psp_ffhq_encode.pt",
            "stylegan2-ffhq-config-f.pt",
            "model_ir_se50.pth",
            "lpips_alex_v0.1.pth",
            "dex_age_classifier.pth",
        ]
        with patch(
            "comfyui_sai_nodes.nodes.image.mytimemachine.folder_paths.get_filename_list",
            return_value=files,
        ):
            self.assertEqual(
                get_checkpoint_options(),
                ["sam_ffhq_aging.pt", "personalized\\iteration_10000.pt"],
            )

    def test_loader_prefers_sam_as_base_model(self) -> None:
        self.assertEqual(
            get_default_base_checkpoint(
                ["iteration_10000.pt", "nested\\sam_ffhq_aging.pt"]
            ),
            "nested\\sam_ffhq_aging.pt",
        )

    def test_checkpoint_role_menus_separate_known_models(self) -> None:
        with (
            patch(
                "comfyui_sai_nodes.nodes.image.mytimemachine.get_checkpoint_options",
                return_value=["sam_ffhq_aging.pt", "iteration_10000.pt"],
            ),
            patch(
                "comfyui_sai_nodes.nodes.image.mytimemachine.is_personalized_checkpoint",
                side_effect=lambda name: name == "iteration_10000.pt",
            ),
        ):
            self.assertEqual(get_base_checkpoint_options(), ["sam_ffhq_aging.pt"])

    def test_checkpoint_roles_match_user_facing_contract(self) -> None:
        self.assertEqual(
            resolve_checkpoint_roles("sam_ffhq_aging.pt", NO_PERSONALIZED_CHECKPOINT),
            ("sam_ffhq_aging.pt", None),
        )
        self.assertEqual(
            resolve_checkpoint_roles("sam_ffhq_aging.pt", "iteration_10000.pt"),
            ("sam_ffhq_aging.pt", "iteration_10000.pt"),
        )

    def test_checkpoint_roles_migrate_former_personalized_order(self) -> None:
        self.assertEqual(
            resolve_checkpoint_roles("iteration_10000.pt", "sam_ffhq_aging.pt"),
            ("sam_ffhq_aging.pt", "iteration_10000.pt"),
        )

    def test_preprocess_preserves_batch_and_adds_normalized_age(self) -> None:
        image = torch.ones((2, 32, 48, 3))
        conditioned = prepare_conditioned_image(image, 35)

        self.assertEqual(tuple(conditioned.shape), (2, 4, 256, 256))
        torch.testing.assert_close(conditioned[:, :3], torch.ones_like(conditioned[:, :3]))
        torch.testing.assert_close(
            conditioned[:, 3], torch.full_like(conditioned[:, 3], 0.35)
        )

    def test_personalized_path_uses_global_latent_and_age(self) -> None:
        local = FakeLocalModel()
        bundle = MyTimeMachineBundle(local, FakeGlobalModel())
        result = run_age_transform(bundle, torch.zeros((2, 4, 256, 256)), 70)

        self.assertEqual(tuple(result.shape), (2, 8, 8, 3))
        torch.testing.assert_close(result, torch.full_like(result, 0.5))
        torch.testing.assert_close(local.blender_ages, torch.tensor([0.7, 0.7]))

    def test_age_range_is_inclusive_and_supports_both_directions(self) -> None:
        self.assertEqual(build_target_ages(42, "single", 10, 30, 1), [42])
        self.assertEqual(build_target_ages(42, "range", 10, 15, 2), [10, 12, 14, 15])
        self.assertEqual(build_target_ages(42, "range", 15, 10, 2), [15, 13, 11, 10])

    def test_age_range_execution_returns_age_major_image_batch(self) -> None:
        bundle = MyTimeMachineBundle(FakeLocalModel(), FakeGlobalModel())

        class FakePatcher:
            model = bundle
            load_device = torch.device("cpu")

            @staticmethod
            def model_dtype():
                return torch.float32

        with (
            patch(
                "comfyui_sai_nodes.nodes.image.mytimemachine.model_management.load_models_gpu"
            ),
            patch(
                "comfyui_sai_nodes.nodes.image.mytimemachine.model_management.intermediate_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "comfyui_sai_nodes.nodes.image.mytimemachine.model_management.throw_exception_if_processing_interrupted"
            ),
            patch("comfyui_sai_nodes.nodes.image.mytimemachine.comfy.utils.ProgressBar"),
        ):
            output = MyTimeMachineAgeTransformSai.execute(
                FakePatcher(),
                torch.zeros((2, 32, 32, 3)),
                30,
                "range",
                10,
                12,
                1,
            )

        self.assertEqual(tuple(output[0].shape), (6, 8, 8, 3))

    def test_portable_stylegan_ops_do_not_require_extension_build(self) -> None:
        activation = FusedLeakyReLU(2)
        output = activation(torch.zeros((1, 2, 4, 4)))
        self.assertEqual(tuple(output.shape), (1, 2, 4, 4))

        filtered = upfirdn2d(torch.ones((1, 2, 4, 4)), torch.ones((2, 2)), up=2)
        self.assertEqual(filtered.ndim, 4)
        self.assertEqual(filtered.shape[1], 2)

    def test_extension_registers_nodes_and_schema(self) -> None:
        nodes = asyncio.run(SaiNodesExtension().get_node_list())
        self.assertIn(LoadMyTimeMachineModelSai, nodes)
        self.assertIn(MyTimeMachineAgeTransformSai, nodes)
        self.assertIn(MyTimeMachineFaceAlignSai, nodes)
        self.assertIn(MyTimeMachineFaceRestoreSai, nodes)

        loader = LoadMyTimeMachineModelSai.define_schema()
        transform = MyTimeMachineAgeTransformSai.define_schema()
        self.assertEqual(loader.node_id, "LoadMyTimeMachineModel_Sai")
        loader_inputs = {item.id: item for item in loader.inputs}
        self.assertEqual(
            loader_inputs["global_checkpoint"].default,
            NO_PERSONALIZED_CHECKPOINT,
        )
        self.assertEqual(loader_inputs["checkpoint"].display_name, "base_checkpoint")
        self.assertEqual(
            loader_inputs["global_checkpoint"].display_name,
            "personalized_checkpoint",
        )
        self.assertEqual(loader.display_name, "Load MyTM Re-ageing Model Ψ")
        self.assertEqual(loader.category, "Sai/Re-ageing/MyTimeMachine")
        self.assertTrue(loader.display_name.endswith("Ψ"))
        self.assertEqual(transform.node_id, "MyTimeMachineAgeTransform_Sai")
        self.assertEqual(transform.display_name, "MyTM Re-ageing Ψ")
        self.assertEqual(transform.category, "Sai/Re-ageing/MyTimeMachine")
        self.assertEqual([output.io_type for output in transform.outputs], ["IMAGE"])
        self.assertEqual(
            [item.id for item in transform.inputs],
            [
                "mytimemachine_model",
                "image",
                "target_age",
                "age_mode",
                "range_start_age",
                "range_end_age",
                "range_step",
            ],
        )


if __name__ == "__main__":
    unittest.main()
