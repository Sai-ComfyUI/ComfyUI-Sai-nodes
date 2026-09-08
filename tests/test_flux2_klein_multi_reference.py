"""Tests for the multi-reference latent node."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_sai_nodes"
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
from comfyui_sai_nodes.nodes.conditioning.flux2_klein_multi_reference import (
    Flux2KleinMultiReferenceLatent,
    scale_image_to_total_pixels,
)


class FakeVae:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.encoded.append(image)
        return image.movedim(-1, 1)[:, :1, ::8, ::8]


class Flux2KleinMultiReferenceTests(unittest.TestCase):
    def test_scale_preserves_ratio_and_rounds_to_resolution_step(self) -> None:
        image = torch.zeros((1, 400, 800, 3), dtype=torch.float32)

        scaled = scale_image_to_total_pixels(image, "nearest-exact", 1.0, 16)

        self.assertEqual(tuple(scaled.shape), (1, 720, 1456, 3))
        self.assertEqual(scaled.shape[1] % 16, 0)
        self.assertEqual(scaled.shape[2] % 16, 0)

    def test_execute_appends_ordered_latents_to_both_conditionings(self) -> None:
        existing = torch.ones((1, 1, 2, 2))
        positive = [[torch.zeros((1, 4)), {"reference_latents": [existing]}]]
        negative = [[torch.zeros((1, 4)), {}]]
        image_1 = torch.full((1, 64, 128, 4), 0.25)
        image_2 = torch.full((1, 128, 64, 3), 0.75)
        vae = FakeVae()

        result = Flux2KleinMultiReferenceLatent.execute(
            positive,
            negative,
            vae,
            "nearest-exact",
            0.01,
            16,
            {"image_2": image_2, "image_1": image_1},
        )
        output_positive, output_negative, first_scaled, output_vae = result.result

        self.assertEqual(len(vae.encoded), 2)
        self.assertTrue(torch.all(vae.encoded[0] == 0.25))
        self.assertTrue(torch.all(vae.encoded[1] == 0.75))
        self.assertEqual(vae.encoded[0].shape[-1], 3)
        self.assertIs(first_scaled, vae.encoded[0])
        self.assertIs(output_vae, vae)
        self.assertEqual(len(output_positive[0][1]["reference_latents"]), 3)
        self.assertIs(output_positive[0][1]["reference_latents"][0], existing)
        self.assertEqual(len(output_negative[0][1]["reference_latents"]), 2)
        self.assertIsNot(output_positive, positive)
        self.assertIsNot(output_negative, negative)

    def test_extension_registers_node(self) -> None:
        nodes = asyncio.run(SaiNodesExtension().get_node_list())
        self.assertIn(Flux2KleinMultiReferenceLatent, nodes)

        schema = Flux2KleinMultiReferenceLatent.define_schema()
        inputs = {input_definition.id: input_definition for input_definition in schema.inputs}
        self.assertEqual(schema.node_id, "Flux2KleinMultiReferenceLatent_Sai")
        self.assertEqual(schema.display_name, "Multi Reference Latent Ψ")
        self.assertEqual(schema.category, "Sai/Conditioning/Edit Model")
        self.assertEqual(
            [output.display_name for output in schema.outputs],
            [
                "positive_conditioning",
                "negative_conditioning",
                "first_image_scaled",
                "vae",
            ],
        )
        self.assertEqual(inputs["resolution_steps"].default, 1)
        self.assertEqual(inputs["images"].template.min, 1)
        self.assertEqual(
            inputs["images"].template.names,
            [f"image_{index}" for index in range(1, 17)],
        )


if __name__ == "__main__":
    unittest.main()
