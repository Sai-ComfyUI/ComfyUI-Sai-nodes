"""Tests for MyTimeMachine automatic face framing and restoration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


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

from comfyui_sai_nodes.nodes.image.face_reframe import (
    ALIGN_SIZE,
    MyTimeMachineFaceAlignSai,
    MyTimeMachineFaceRestoreSai,
    align_face,
    calculate_ffhq_quad,
    make_blend_mask,
    restore_face,
)


SYNTHETIC_LANDMARKS = np.asarray(
    [
        [42.0, 42.0],
        [82.0, 42.0],
        [62.0, 61.0],
        [48.0, 79.0],
        [76.0, 79.0],
    ],
    dtype=np.float32,
)


class FaceReframeTests(unittest.TestCase):
    def test_quad_scale_preserves_center_and_changes_size(self) -> None:
        normal = calculate_ffhq_quad(SYNTHETIC_LANDMARKS, 1.0)
        wider = calculate_ffhq_quad(SYNTHETIC_LANDMARKS, 1.2)

        np.testing.assert_allclose(normal.mean(axis=0), wider.mean(axis=0), atol=1e-5)
        self.assertGreater(np.linalg.norm(wider[2] - wider[0]), np.linalg.norm(normal[2] - normal[0]))

    def test_masks_have_expected_shape_and_soft_range(self) -> None:
        for shape in ("oval (recommended)", "full crop"):
            mask = make_blend_mask(shape, 1.0, 0.12)
            self.assertEqual(mask.shape, (ALIGN_SIZE, ALIGN_SIZE))
            self.assertGreaterEqual(float(mask.min()), 0.0)
            self.assertLessEqual(float(mask.max()), 1.0)
            self.assertGreater(float(mask[ALIGN_SIZE // 2, ALIGN_SIZE // 2]), 0.99)

    def test_align_restore_identity_keeps_source(self) -> None:
        yy, xx = np.mgrid[0:128, 0:128]
        source = np.stack(
            [xx * 2, yy * 2, (xx + yy)],
            axis=-1,
        ).clip(0, 255).astype(np.uint8)
        quad = calculate_ffhq_quad(SYNTHETIC_LANDMARKS, 1.0)
        aligned, inverse = align_face(source, quad)
        restored, mask = restore_face(
            source,
            aligned,
            aligned,
            inverse,
            "oval (recommended)",
            1.0,
            0.12,
            0.0,
        )

        self.assertEqual(restored.shape, (128, 128, 3))
        self.assertEqual(mask.shape, (128, 128))
        active = mask > 0.5
        expected = source.astype(np.float32) / 255.0
        self.assertLess(float(np.abs(restored[active] - expected[active]).mean()), 0.015)

    def test_nodes_support_batches_and_context(self) -> None:
        image = torch.zeros((2, 128, 128, 3), dtype=torch.float32)
        image[:, 24:104, 24:104] = 0.6
        with patch(
            "comfyui_sai_nodes.nodes.image.face_reframe.detect_face_landmarks",
            return_value=SYNTHETIC_LANDMARKS,
        ):
            aligned_output = MyTimeMachineFaceAlignSai.execute(image, 0, 1.0, 0.8)
        aligned, context = aligned_output.result
        self.assertEqual(tuple(aligned.shape), (2, ALIGN_SIZE, ALIGN_SIZE, 3))

        restored_output = MyTimeMachineFaceRestoreSai.execute(
            aligned,
            context,
            "oval (recommended)",
            1.0,
            0.12,
            0.0,
        )
        composite, mask = restored_output.result
        self.assertEqual(tuple(composite.shape), (2, 128, 128, 3))
        self.assertEqual(tuple(mask.shape), (2, 128, 128))

    def test_node_schemas(self) -> None:
        align_schema = MyTimeMachineFaceAlignSai.define_schema()
        restore_schema = MyTimeMachineFaceRestoreSai.define_schema()
        self.assertEqual(align_schema.node_id, "MyTimeMachineFaceAlign_Sai")
        self.assertEqual(restore_schema.node_id, "MyTimeMachineFaceRestore_Sai")
        self.assertEqual(align_schema.display_name, "YuNet FFHQ Face Align Ψ")
        self.assertEqual(restore_schema.display_name, "YuNet FFHQ Face Restore Ψ")
        self.assertEqual(
            align_schema.category,
            "Sai/Face Alignment/YuNet FFHQ",
        )
        self.assertEqual([output.io_type for output in restore_schema.outputs], ["IMAGE", "MASK"])


if __name__ == "__main__":
    unittest.main()
