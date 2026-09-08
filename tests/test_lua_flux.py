"""Tests for the LUA-FLUX latent upscaler nodes."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from comfyui_sai_nodes.nodes.latent.lua_flux import (
    DOWNLOAD_OPTION,
    LoadLuaFluxModel,
    LuaFluxLatentUpscale,
    _FLUX_MODEL_CONFIG,
    extract_lua_state_dict,
    upscale_lua_latent,
)
from comfyui_sai_nodes.nodes.latent.swinir_arch import SwinIRMultiHead


class TinyLuaModel(SwinIRMultiHead):
    """Small test double that still satisfies the node's model type check."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.window_size = 4
        self.head_scales = {"x2": 2.0, "x4": 4.0}
        self.weight = nn.Parameter(torch.ones(()))
        self.last_input_shape: tuple[int, ...] | None = None

    def forward_single_head(self, x: torch.Tensor, head_name: str) -> torch.Tensor:
        self.last_input_shape = tuple(x.shape)
        return F.interpolate(x, scale_factor=int(self.head_scales[head_name]), mode="nearest")


class FakePatcher:
    def __init__(self, model: TinyLuaModel) -> None:
        self.model = model
        self.load_device = torch.device("cpu")

    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype


class LuaFluxTests(unittest.TestCase):
    def test_official_pad_and_crop_sequence_restores_exact_scale(self) -> None:
        model = TinyLuaModel()
        latent = torch.arange(1 * 16 * 5 * 6, dtype=torch.float32).reshape(
            1, 16, 5, 6
        )

        output = upscale_lua_latent(model, latent, "x2")

        self.assertEqual(model.last_input_shape, (1, 16, 8, 8))
        self.assertEqual(tuple(output.shape), (1, 16, 10, 12))

    def test_rejects_flux2_channel_count(self) -> None:
        model = TinyLuaModel()
        with self.assertRaisesRegex(ValueError, "FLUX.2"):
            upscale_lua_latent(model, torch.zeros((1, 128, 8, 8)), "x2")

    def test_execute_preserves_latent_metadata_and_removes_noise_mask(self) -> None:
        model = TinyLuaModel()
        patcher = FakePatcher(model)
        latent = {
            "samples": torch.zeros((1, 16, 8, 8)),
            "noise_mask": torch.ones((1, 8, 8)),
            "batch_index": [7],
        }

        with (
            patch(
                "comfyui_sai_nodes.nodes.latent.lua_flux.model_management.load_models_gpu"
            ) as load_models_gpu,
            patch(
                "comfyui_sai_nodes.nodes.latent.lua_flux.model_management.intermediate_device",
                return_value=torch.device("cpu"),
            ),
        ):
            output = LuaFluxLatentUpscale.execute(patcher, latent, "x4").result[0]

        load_models_gpu.assert_called_once()
        self.assertEqual(tuple(output["samples"].shape), (1, 16, 32, 32))
        self.assertEqual(output["batch_index"], [7])
        self.assertNotIn("noise_mask", output)
        self.assertIsNot(output, latent)

    def test_extracts_ema_checkpoint(self) -> None:
        tensor = torch.ones((1,))
        state_dict = extract_lua_state_dict(
            {"params": {"fallback": tensor}, "params_ema": {"preferred": tensor}}
        )
        self.assertEqual(list(state_dict), ["preferred"])

    def test_official_model_constructs_without_full_weight_allocation(self) -> None:
        with torch.device("meta"):
            model = SwinIRMultiHead(**_FLUX_MODEL_CONFIG)

        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
        self.assertEqual(model.head_scales, {"x2": 2.0, "x4": 4.0})

    def test_extension_registers_lua_nodes_and_schema(self) -> None:
        nodes = asyncio.run(SaiNodesExtension().get_node_list())
        self.assertIn(LoadLuaFluxModel, nodes)
        self.assertIn(LuaFluxLatentUpscale, nodes)

        loader_schema = LoadLuaFluxModel.define_schema()
        upscale_schema = LuaFluxLatentUpscale.define_schema()
        loader_inputs = {item.id: item for item in loader_schema.inputs}
        upscale_inputs = {item.id: item for item in upscale_schema.inputs}

        self.assertEqual(loader_schema.node_id, "LoadLuaFluxModel_Sai")
        self.assertEqual(loader_inputs["model_name"].default, DOWNLOAD_OPTION)
        self.assertEqual(upscale_schema.node_id, "LuaFluxLatentUpscale_Sai")
        self.assertEqual(upscale_inputs["scale"].options, ["x2", "x4"])


if __name__ == "__main__":
    unittest.main()
