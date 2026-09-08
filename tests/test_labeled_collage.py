"""Tests for Labeled Image Collage Ψ."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from PIL import Image


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

from comfyui_sai_nodes.nodes.image.labeled_collage import (
    AUTO_FONT,
    LabeledImageCollage,
    compose_collage,
    draw_label,
    font_options,
    label_extension_size,
    load_font,
    prepare_images,
)


def execute_node(images, labels="A\nB", **overrides):
    options = {
        "layout": "horizontal",
        "items_per_line": 32,
        "size_mode": "uniform cells (fit)",
        "cell_width": 100,
        "cell_height": 80,
        "cross_axis_size": 64,
        "spacing": 4,
        "outer_margin": 5,
        "background_color": "#102030",
        "label_placement": "overlay",
        "label_position": "top center",
        "text_direction": "horizontal",
        "font_size": 20,
        "font_color": "#ffffffff",
        "label_background_color": "#000000ff",
        "label_background_extent": "text box",
        "label_padding": 3,
        "label_margin": 2,
        "outline_color": "#000000ff",
        "outline_width": 1,
        "font_name": AUTO_FONT,
        "font_path": "",
    }
    options.update(overrides)
    return LabeledImageCollage.execute(images=images, labels=labels, **options).result[0]


class LabeledCollageTests(unittest.TestCase):
    def test_schema_and_registration_contract(self) -> None:
        schema = LabeledImageCollage.define_schema()
        self.assertEqual(schema.node_id, "LabeledImageCollage_Sai")
        self.assertEqual(schema.display_name, "Labeled Image Collage Ψ")
        self.assertEqual(schema.category, "Sai/Image/Collage")
        self.assertEqual([output.io_type for output in schema.outputs], ["IMAGE"])
        input_ids = [input_.id for input_ in schema.inputs]
        self.assertIn("font_name", input_ids)
        self.assertIn("label_placement", input_ids)
        self.assertIn("items_per_line", input_ids)
        self.assertNotIn("font_index", input_ids)
        self.assertNotIn("label_background_opacity", input_ids)
        self.assertTrue(any("NotoSansMonoCJK-VF.ttf.ttc" in option for option in font_options()))

    def test_uniform_horizontal_dimensions_and_background(self) -> None:
        red = torch.zeros((1, 20, 40, 3), dtype=torch.float32)
        red[..., 0] = 1.0
        green_batch = torch.zeros((2, 40, 20, 3), dtype=torch.float32)
        green_batch[..., 1] = 1.0
        result = execute_node({"image_1": red, "image_2": green_batch}, labels="")

        self.assertEqual(tuple(result.shape), (1, 90, 318, 3))
        expected = torch.tensor([0x10, 0x20, 0x30], dtype=torch.float32) / 255.0
        torch.testing.assert_close(result[0, 0, 0], expected)

    def test_match_cross_axis_dimensions(self) -> None:
        wide = Image.new("RGB", (80, 40), "red")
        tall = Image.new("RGB", (20, 40), "green")
        horizontal = prepare_images([wide, tall], "horizontal", "match cross-axis", 99, 99, 60, (0, 0, 0, 255))
        self.assertEqual([image.size for image in horizontal], [(120, 60), (30, 60)])
        vertical = prepare_images([wide, tall], "vertical", "match cross-axis", 99, 99, 50, (0, 0, 0, 255))
        self.assertEqual([image.size for image in vertical], [(50, 25), (50, 100)])
        collage = compose_collage(vertical, "vertical", 7, 3, (0, 0, 0, 255))
        self.assertEqual(collage.size, (56, 138))

    def test_label_position_and_vertical_text_change_pixels(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        font = load_font(AUTO_FONT, "", 20)
        labeled = draw_label(
            image,
            "AB",
            font,
            "vertical",
            "bottom right",
            "overlay",
            (255, 0, 0, 255),
            (0, 0, 0, 255),
            "text box",
            4,
            3,
            (0, 0, 0, 255),
            0,
        )
        array = np.asarray(labeled)
        self.assertTrue(np.all(array[0, 0] == 255))
        self.assertGreater(int(np.count_nonzero(np.any(array[50:, 60:] != 255, axis=-1))), 100)

    def test_traditional_chinese_font_renders(self) -> None:
        noto = PACKAGE_ROOT / "temp" / "NotoSansMonoCJK-VF.ttf.ttc"
        if not noto.is_file():
            self.skipTest("User-provided Noto CJK test font is unavailable.")
        font = load_font(AUTO_FONT, str(noto), 36)
        image = Image.new("RGB", (240, 100), "white")
        chinese_text = "\u7e41\u9ad4\u4e2d\u6587\u6e2c\u8a66"
        labeled = draw_label(
            image,
            chinese_text,
            font,
            "horizontal",
            "top center",
            "overlay",
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            "text box",
            2,
            0,
            (255, 255, 255, 255),
            0,
        )
        question_marks = draw_label(
            image,
            "?" * len(chinese_text),
            font,
            "horizontal",
            "top center",
            "overlay",
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            "text box",
            2,
            0,
            (255, 255, 255, 255),
            0,
        )
        rendered = np.asarray(labeled)
        self.assertGreater(int(np.count_nonzero(np.any(rendered < 128, axis=-1))), 500)
        self.assertFalse(np.array_equal(rendered, np.asarray(question_marks)))

    def test_picker_alpha_and_full_width_banner_are_applied(self) -> None:
        image = Image.new("RGB", (160, 100), "white")
        font = load_font(AUTO_FONT, "", 24)
        labeled = draw_label(
            image,
            "A",
            font,
            "horizontal",
            "top left",
            "overlay",
            (0, 0, 0, 128),
            (0, 0, 0, 128),
            "full width banner",
            4,
            0,
            (0, 0, 0, 0),
            0,
        )
        array = np.asarray(labeled)
        self.assertTrue(np.all(np.abs(array[0, -1].astype(int) - 127) <= 1))
        self.assertTrue(np.all(array[-1, -1] == 255))
        self.assertGreater(int(array.min()), 0)

    def test_full_height_banner_fills_vertical_axis_only(self) -> None:
        image = Image.new("RGB", (160, 100), "white")
        font = load_font(AUTO_FONT, "", 24)
        labeled = draw_label(
            image,
            "A",
            font,
            "horizontal",
            "top left",
            "overlay",
            (255, 255, 255, 255),
            (255, 0, 0, 255),
            "full height banner",
            4,
            0,
            (0, 0, 0, 0),
            0,
        )
        array = np.asarray(labeled)
        self.assertTrue(np.all(array[-1, 0] == np.asarray([255, 0, 0])))
        self.assertTrue(np.all(array[-1, -1] == 255))

    def test_extend_top_preserves_every_original_pixel(self) -> None:
        source = np.arange(60 * 80 * 3, dtype=np.uint8).reshape((60, 80, 3))
        image = Image.fromarray(source, "RGB")
        font = load_font(AUTO_FONT, "", 18)
        extension = label_extension_size(image, "Label", font, "horizontal", "extend top", 4, 3, 1)
        labeled = draw_label(
            image,
            "Label",
            font,
            "horizontal",
            "top center",
            "extend top",
            (255, 255, 255, 255),
            (12, 34, 56, 255),
            "text box",
            4,
            3,
            (0, 0, 0, 255),
            1,
            extension,
        )

        self.assertEqual(labeled.size, (80, 60 + extension))
        np.testing.assert_array_equal(np.asarray(labeled)[extension:, :, :], source)

    def test_all_extend_directions_change_only_the_requested_axis(self) -> None:
        image = Image.new("RGB", (90, 60), (20, 40, 60))
        font = load_font(AUTO_FONT, "", 18)
        for placement in ("extend top", "extend bottom", "extend left", "extend right"):
            with self.subTest(placement=placement):
                extension = label_extension_size(image, "A", font, "horizontal", placement, 4, 2, 0)
                labeled = draw_label(
                    image,
                    "A",
                    font,
                    "horizontal",
                    "bottom right",
                    placement,
                    (255, 255, 255, 255),
                    (0, 0, 0, 255),
                    "full width banner",
                    4,
                    2,
                    (0, 0, 0, 255),
                    0,
                    extension,
                )
                expected = (90, 60 + extension) if placement.endswith(("top", "bottom")) else (90 + extension, 60)
                self.assertEqual(labeled.size, expected)

    def test_execute_uses_one_shared_extension_size_for_all_cells(self) -> None:
        red = torch.zeros((1, 40, 60, 3), dtype=torch.float32)
        red[..., 0] = 1.0
        result = execute_node(
            {"image_1": red, "image_2": red.clone()},
            labels="A\nA much longer label that wraps",
            label_placement="extend top",
            spacing=0,
            outer_margin=0,
        )

        self.assertEqual(result.shape[2], 200)
        self.assertGreater(result.shape[1], 80)

    def test_horizontal_layout_wraps_and_pads_the_last_row(self) -> None:
        colors = ["red", "green", "blue", "yellow", "magenta"]
        images = [Image.new("RGB", (20, 10), color) for color in colors]
        collage = compose_collage(images, "horizontal", 2, 3, (1, 2, 3, 255), items_per_line=3)

        self.assertEqual(collage.size, (70, 28))
        array = np.asarray(collage)
        np.testing.assert_array_equal(array[20, 55], np.asarray([1, 2, 3]))
        np.testing.assert_array_equal(array[5, 5], np.asarray([255, 0, 0]))
        np.testing.assert_array_equal(array[17, 5], np.asarray([255, 255, 0]))

    def test_vertical_layout_wraps_and_pads_the_last_column(self) -> None:
        colors = ["red", "green", "blue", "yellow", "magenta"]
        images = [Image.new("RGB", (20, 10), color) for color in colors]
        collage = compose_collage(images, "vertical", 2, 3, (1, 2, 3, 255), items_per_line=3)

        self.assertEqual(collage.size, (48, 40))
        array = np.asarray(collage)
        np.testing.assert_array_equal(array[30, 30], np.asarray([1, 2, 3]))
        np.testing.assert_array_equal(array[5, 5], np.asarray([255, 0, 0]))
        np.testing.assert_array_equal(array[5, 27], np.asarray([255, 255, 0]))

    def test_background_picker_alpha_produces_rgba_output(self) -> None:
        red = torch.zeros((1, 20, 40, 3), dtype=torch.float32)
        red[..., 0] = 1.0
        result = execute_node(
            {"image_1": red},
            labels="",
            background_color="#10203040",
        )

        self.assertEqual(tuple(result.shape), (1, 90, 110, 4))
        self.assertAlmostEqual(float(result[0, 0, 0, 3]), 0x40 / 255.0)
        self.assertAlmostEqual(float(result[0, 45, 55, 3]), 1.0)

    def test_requires_an_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "Connect at least one image"):
            execute_node({}, labels="")

    def test_named_preset_frontend_contract(self) -> None:
        source = (PACKAGE_ROOT / "web" / "labeled_collage_presets_v2.js").read_text(encoding="utf-8")
        self.assertIn('const NODE_ID = "LabeledImageCollage_Sai"', source)
        self.assertIn('"items_per_line"', source)
        self.assertIn('"save preset"', source)
        self.assertIn('"delete preset"', source)
        self.assertIn('"labels",', source)


if __name__ == "__main__":
    unittest.main()
