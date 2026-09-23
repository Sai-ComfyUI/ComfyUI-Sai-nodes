"""Contract and math tests for the native LumaFlux nodes."""

from __future__ import annotations

import asyncio
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_sai_nodes"
if PACKAGE_NAME not in sys.modules:
    package_spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    assert package_spec is not None and package_spec.loader is not None
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[PACKAGE_NAME] = package
    package_spec.loader.exec_module(package)

from comfyui_sai_nodes import SaiNodesExtension
from comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux import (
    ApplyLumaFluxAdapter,
    HDR_FORMAT_EXR,
    HDR_FORMAT_PNG,
    PREVIEW_METHODS,
    RESIZE_MODES,
    HdrToSdrPreview,
    LumaFluxSdrToHdr,
    LumaFluxModelHandle,
    SaveHdrImage,
    _save_exr_float32,
    _save_png16_pq,
    hdr_pq_to_linear_relative,
    nearest_multiple,
    next_multiple,
    normalize_boolean_input,
    prepare_lumaflux_image,
    resolve_lumaflux_geometry,
    time_shift,
    validate_hdr_image,
)
from comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video import (
    HDR10VideoWriter,
    HDR_EXR_SEQUENCE_FORMAT,
    HDR_PNG_SEQUENCE_FORMAT,
    HDR_VIDEO_FORMAT,
    HDRVideoStream,
    LumaFluxSdrVideoToHdr,
    SaveHdr10Video,
)
from comfyui_sai_nodes.vendor.lumaflux_native.adapter import (
    LumaFluxAdapter,
    RQSToneDecoder,
    remap_checkpoint_keys,
)
from comfyui_sai_nodes.vendor.lumaflux_native.color import (
    hdr_to_sdr_bt2446c,
    hdr_to_sdr_whitepoint,
    pq_oetf_nits,
)


class LumaFluxTests(unittest.TestCase):
    def test_checkpoint_key_mapping_matches_native_adapter(self) -> None:
        state = {
            "transformer.transformer_blocks.0.modulation.head.bias": torch.zeros(6),
            "transformer.transformer_blocks.3.pga.down.weight": torch.zeros(8, 3072),
            "transformer.single_transformer_blocks.4.pcm.mlp.0.bias": torch.zeros(64),
            "null_pooled": torch.zeros(768),
        }
        mapped = remap_checkpoint_keys(state)
        self.assertEqual(
            set(mapped),
            {
                "modulation.head.bias",
                "double_blocks.3.pga.down.weight",
                "single_blocks.4.pcm.mlp.0.bias",
                "null_pooled",
            },
        )

    def test_adapter_contract_has_released_parameter_count(self) -> None:
        with torch.device("meta"):
            adapter = LumaFluxAdapter()
        self.assertEqual(len(dict(adapter.named_parameters())), 831)
        self.assertEqual(len(adapter.double_blocks), 19)
        self.assertEqual(len(adapter.single_blocks), 38)

    def test_time_shift_identity_and_endpoints(self) -> None:
        values = torch.tensor([1.0, 0.5, 0.0])
        self.assertTrue(torch.equal(time_shift(values, 1.0), values))
        shifted = time_shift(values, 2.0)
        self.assertEqual(shifted[0].item(), 1.0)
        self.assertEqual(shifted[-1].item(), 0.0)

    def test_nearest_multiple_of_16_chooses_nearest_and_ties_up(self) -> None:
        self.assertEqual(nearest_multiple(1024), 1024)
        self.assertEqual(nearest_multiple(1000), 1008)
        self.assertEqual(nearest_multiple(1016), 1024)
        self.assertEqual(nearest_multiple(7), 16)

    def test_boolean_widget_values_are_normalized_without_string_truthiness(self) -> None:
        self.assertTrue(normalize_boolean_input(True, "shared_noise"))
        self.assertTrue(normalize_boolean_input("true", "shared_noise"))
        self.assertTrue(normalize_boolean_input("Temporal consistency on", "shared_noise"))
        self.assertFalse(normalize_boolean_input(False, "shared_noise"))
        self.assertFalse(normalize_boolean_input("false", "shared_noise"))
        self.assertFalse(normalize_boolean_input("Independent frames", "shared_noise"))
        with self.assertRaisesRegex(ValueError, "shared_noise must be a boolean"):
            normalize_boolean_input("unexpected", "shared_noise")

    def test_geometry_modes_pad_crop_resize_or_validate(self) -> None:
        image = torch.rand(1, 777, 1000, 3)
        padded, crop_size, notice = prepare_lumaflux_image(image, "crop")
        self.assertEqual(tuple(padded.shape), (1, 784, 1008, 3))
        self.assertEqual(crop_size, (1000, 777))
        self.assertIn("edge-padded 1000×777 → 1008×784", notice)
        self.assertTrue(torch.equal(padded[:, :777, :1000], image))
        self.assertTrue(torch.equal(padded[:, -1, -1], image[:, -1, -1]))
        resized, output_size, notice = prepare_lumaflux_image(image, "resize")
        self.assertEqual(tuple(resized.shape), (1, 784, 1008, 3))
        self.assertEqual(output_size, (1008, 784))
        self.assertIn("resized 1000×777 → 1008×784", notice)
        with self.assertRaisesRegex(ValueError, "Select crop"):
            prepare_lumaflux_image(image, "none")
        self.assertEqual(next_multiple(1000), 1008)
        self.assertEqual(resolve_lumaflux_geometry(1000, 777, "crop")[1], (1000, 777))

    def test_bt2446_preview_maps_mastering_white_to_sdr(self) -> None:
        white_pq = pq_oetf_nits(torch.tensor(1000.0))
        hdr = white_pq.expand(1, 3, 2, 2)
        preview = hdr_to_sdr_bt2446c(hdr, 1000.0)
        self.assertTrue(torch.allclose(preview, torch.ones_like(preview), atol=2e-3))

    def test_whitepoint_preview_supports_reinhard_and_clip(self) -> None:
        hdr = torch.rand(1, 3, 8, 8) * pq_oetf_nits(torch.tensor(1000.0))
        for method in ("reinhard", "clip"):
            preview = hdr_to_sdr_whitepoint(hdr, 203.0, 1000.0, method)
            self.assertEqual(tuple(preview.shape), tuple(hdr.shape))
            self.assertGreaterEqual(preview.min().item(), 0.0)
            self.assertLessEqual(preview.max().item(), 1.0)

    def test_exr_conversion_is_linear_and_peak_normalized(self) -> None:
        mastering_white = pq_oetf_nits(torch.tensor(1000.0))
        pq = torch.stack([torch.tensor(0.0), mastering_white]).reshape(1, 1, 2, 1)
        linear = hdr_pq_to_linear_relative(pq, 1000.0)
        self.assertLess(linear[0, 0, 0, 0].item(), 1e-7)
        self.assertAlmostEqual(linear[0, 0, 1, 0].item(), 1.0, places=4)

    def test_exr_writer_embeds_hdr_color_attributes(self) -> None:
        peak_pq = pq_oetf_nits(torch.tensor(1000.0))
        image = peak_pq.expand(2, 3, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.exr"
            _save_exr_float32(image, str(path), 1000.0)
            data = path.read_bytes()
        self.assertEqual(data[:4], (20000630).to_bytes(4, "little"))
        self.assertIn(b"chromaticities\0", data)
        self.assertIn(b"adoptedNeutral\0", data)
        self.assertIn(b"whiteLuminance\0", data)
        self.assertIn(b"Linear BT.2020 RGB", data)

    def test_png16_writer_preserves_values_and_embeds_pq_bt2020_cicp(self) -> None:
        image = torch.rand(5, 7, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.png"
            _save_png16_pq(image, str(path))
            data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn(b"cICP" + bytes([9, 16, 0, 1]), data)

    def test_hdr_contract_rejects_plain_image(self) -> None:
        with self.assertRaisesRegex(TypeError, "HDR_IMAGE"):
            validate_hdr_image(torch.zeros(1, 2, 2, 3))
        with self.assertRaisesRegex(ValueError, "PQ"):
            validate_hdr_image(
                {"images": torch.zeros(1, 2, 2, 3), "transfer": "srgb", "primaries": "bt709"}
            )

    def test_rqs_accepts_applied_spline_parameters_for_video_ema(self) -> None:
        decoder = RQSToneDecoder()
        decoded = torch.rand(1, 3, 16, 16) * decoder.peak_pq
        latent = torch.rand(1, 16, 2, 2)
        current = decoder.spline_params(latent)
        explicit = decoder(decoded, latent, 1.0, spline_params=current)
        implicit = decoder(decoded, latent, 1.0)
        self.assertTrue(torch.allclose(explicit, implicit))

    def test_hdr_video_contract_is_lazy_and_engine_independent(self) -> None:
        calls = []

        def frames():
            calls.append("started")
            yield torch.zeros(1, 16, 16, 3)

        stream = HDRVideoStream(frames, Fraction(24, 1), 1, (16, 16), (16, 16), object())
        self.assertEqual(calls, [])
        self.assertEqual(tuple(next(stream.iter_frames()).shape), (1, 16, 16, 3))
        self.assertEqual(calls, ["started"])

    def test_video_solver_reuses_noise_and_previous_applied_spline_state(self) -> None:
        class Video:
            @staticmethod
            def get_dimensions(): return (16, 16)
            @staticmethod
            def get_frame_rate(): return Fraction(24, 1)
            @staticmethod
            def get_frame_count(): return 2

        handle = LumaFluxModelHandle(None, None, None, None, None, "test")
        calls = []

        def fake_convert(*_args, **kwargs):
            calls.append((kwargs["noise"], kwargs["previous_spline_params"]))
            index = len(calls)
            return {
                "images": torch.zeros(1, 16, 16, 3),
                "bridge_noise": torch.full((1, 16, 2, 2), 7.0),
                "spline_params": {
                    "widths": torch.full((1, 8), float(index)),
                    "heights": torch.full((1, 8), float(index)),
                    "derivs": torch.full((1, 9), float(index)),
                },
            }

        with mock.patch(
            "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video.iter_video_frames",
            return_value=iter([torch.zeros(1, 16, 16, 3), torch.zeros(1, 16, 16, 3)]),
        ), mock.patch(
            "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video.convert_sdr_to_hdr",
            side_effect=fake_convert,
        ):
            output = LumaFluxSdrVideoToHdr.execute(
                handle, Video(), "crop", 1, 0, True, 0.8, 1.0, 0.05, 1.0
            )
            list(output[0].iter_frames())

        self.assertIsNone(calls[0][0])
        self.assertIsNone(calls[0][1])
        self.assertTrue(torch.equal(calls[1][0], torch.full((1, 16, 2, 2), 7.0)))
        self.assertTrue(torch.equal(calls[1][1]["widths"], torch.ones(1, 8)))

    def test_image_batch_uses_the_same_temporal_consistency_contract(self) -> None:
        handle = LumaFluxModelHandle(None, None, None, None, None, "test")
        calls = []

        def fake_convert(*_args, **kwargs):
            calls.append((kwargs["noise"], kwargs["previous_spline_params"], kwargs["spline_ema"]))
            index = len(calls)
            return {
                "images": torch.full((1, 16, 16, 3), float(index)),
                "bridge_noise": torch.full((1, 16, 2, 2), 5.0),
                "spline_params": {
                    "widths": torch.full((1, 8), float(index)),
                    "heights": torch.full((1, 8), float(index)),
                    "derivs": torch.full((1, 9), float(index)),
                },
                "resize_notice": "Size unchanged",
            }

        with mock.patch(
            "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux.convert_sdr_to_hdr",
            side_effect=fake_convert,
        ):
            output = LumaFluxSdrToHdr.execute(
                handle, torch.zeros(2, 16, 16, 3), "none", 1, 0, True, 0.8, 1.0, 0.05, 1.0
            )

        self.assertEqual(tuple(output[0]["images"].shape), (2, 16, 16, 3))
        self.assertIsNone(calls[0][0])
        self.assertIsNone(calls[0][1])
        self.assertTrue(torch.equal(calls[1][0], torch.full((1, 16, 2, 2), 5.0)))
        self.assertTrue(torch.equal(calls[1][1]["widths"], torch.ones(1, 8)))
        self.assertEqual(calls[1][2], 0.8)

    def test_video_solver_disables_all_cross_frame_state_with_shared_noise_off(self) -> None:
        class Video:
            @staticmethod
            def get_dimensions(): return (16, 16)
            @staticmethod
            def get_frame_rate(): return Fraction(24, 1)
            @staticmethod
            def get_frame_count(): return 2

        handle = LumaFluxModelHandle(None, None, None, None, None, "test")
        calls = []

        def fake_convert(*args, **kwargs):
            calls.append((args[3], kwargs["noise"], kwargs["previous_spline_params"], kwargs["spline_ema"]))
            return {
                "images": torch.zeros(1, 16, 16, 3),
                "bridge_noise": torch.ones(1, 16, 2, 2),
                "spline_params": {"widths": torch.ones(1, 8), "heights": torch.ones(1, 8), "derivs": torch.ones(1, 9)},
            }

        with mock.patch(
            "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video.iter_video_frames",
            return_value=iter([torch.zeros(1, 16, 16, 3), torch.zeros(1, 16, 16, 3)]),
        ), mock.patch(
            "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video.convert_sdr_to_hdr",
            side_effect=fake_convert,
        ):
            output = LumaFluxSdrVideoToHdr.execute(
                handle, Video(), "none", 1, 10, False, 0.8, 1.0, 0.05, 1.0
            )
            list(output[0].iter_frames())

        self.assertEqual([call[0] for call in calls], [10, 11])
        self.assertTrue(all(call[1] is None and call[2] is None and call[3] == 0.0 for call in calls))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is unavailable")
    def test_hdr10_writer_emits_main10_pq_bt2020_and_mastering_metadata(self) -> None:
        class Source:
            @staticmethod
            def get_stream_source():
                return "unused.mp4"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hdr.mp4"
            writer = HDR10VideoWriter(path, 16, 16, Fraction(24, 1), 28, "fast", 1000.0, Source(), False)
            writer.write(torch.zeros(1, 16, 16, 3))
            writer.write(torch.full((1, 16, 16, 3), 0.5))
            writer.close()
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_streams", "-show_frames", "-read_intervals", "%+#1", "-of", "json", str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        metadata = json.loads(probe.stdout)
        stream = metadata["streams"][0]
        self.assertEqual(stream["pix_fmt"], "yuv420p10le")
        self.assertEqual(stream["color_space"], "bt2020nc")
        self.assertEqual(stream["color_transfer"], "smpte2084")
        self.assertEqual(stream["color_primaries"], "bt2020")
        side_data = json.dumps(metadata.get("frames", []))
        self.assertIn("Mastering display metadata", side_data)
        self.assertIn("Content light level metadata", side_data)

    def test_stream_saver_writes_exr_and_png_sequences_without_materializing_batch(self) -> None:
        for format_name, extension in (
            (HDR_EXR_SEQUENCE_FORMAT, "exr"),
            (HDR_PNG_SEQUENCE_FORMAT, "png"),
        ):
            calls = []

            def frames():
                for index in range(2):
                    calls.append(index)
                    yield torch.full((1, 16, 16, 3), index * 0.25)

            stream = HDRVideoStream(frames, Fraction(24, 1), 2, (16, 16), (16, 16), object())
            with tempfile.TemporaryDirectory() as directory, mock.patch(
                "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video.folder_paths.get_output_directory",
                return_value=directory,
            ), mock.patch(
                "comfyui_sai_nodes.nodes.sdr_to_hdr.lumaflux_video.folder_paths.get_save_image_path",
                return_value=(directory, "clip", 1, "", "clip"),
            ):
                SaveHdr10Video.execute(stream, "clip", format_name, 16, "fast", "No audio")
                files = sorted(Path(directory, "clip_00001").glob(f"*.{extension}"))
                self.assertEqual(len(files), 2)
                self.assertEqual(len(list(Path(directory, "clip_00001").glob("*.json"))), 2)
                if extension == "png":
                    self.assertIn(b"cICP" + bytes([9, 16, 0, 1]), files[0].read_bytes())
                else:
                    self.assertEqual(files[0].read_bytes()[:4], (20000630).to_bytes(4, "little"))
            self.assertEqual(calls, [0, 1])

    def test_extension_registers_nodes_and_stable_custom_types(self) -> None:
        nodes = asyncio.run(SaiNodesExtension().get_node_list())
        self.assertIn(ApplyLumaFluxAdapter, nodes)
        self.assertIn(LumaFluxSdrToHdr, nodes)
        self.assertIn(LumaFluxSdrVideoToHdr, nodes)
        self.assertIn(HdrToSdrPreview, nodes)
        self.assertIn(SaveHdrImage, nodes)
        self.assertIn(SaveHdr10Video, nodes)
        loader, converter, preview = (
            ApplyLumaFluxAdapter.define_schema(),
            LumaFluxSdrToHdr.define_schema(),
            HdrToSdrPreview.define_schema(),
        )
        self.assertEqual(loader.node_id, "ApplyLumaFluxAdapter_Sai")
        self.assertEqual(converter.outputs[0].io_type, "HDR_IMAGE")
        self.assertEqual(preview.inputs[0].io_type, "HDR_IMAGE")
        self.assertNotIn("SAI_", converter.outputs[0].io_type)
        saver = SaveHdrImage.define_schema()
        self.assertEqual(saver.category, "Sai/Saver/HDR")
        self.assertTrue(saver.is_output_node)
        self.assertTrue(loader.display_name.endswith("Ψ"))
        self.assertTrue(converter.display_name.endswith("Ψ"))
        self.assertTrue(preview.display_name.endswith("Ψ"))
        self.assertTrue(saver.display_name.endswith("Ψ"))
        converter_inputs = {value.id: value for value in converter.inputs}
        self.assertEqual(converter_inputs["resize_mode"].default, "crop")
        self.assertEqual(converter_inputs["resize_mode"].options, RESIZE_MODES)
        self.assertTrue(converter_inputs["shared_noise"].default)
        self.assertIsNone(getattr(converter_inputs["shared_noise"], "label_on", None))
        self.assertIsNone(getattr(converter_inputs["shared_noise"], "label_off", None))
        preview_inputs = {value.id: value for value in preview.inputs}
        self.assertEqual(preview_inputs["method"].options, PREVIEW_METHODS)
        format_input = next(value for value in saver.inputs if value.id == "format")
        self.assertEqual(format_input.default, HDR_FORMAT_EXR)
        self.assertEqual(format_input.options, [HDR_FORMAT_EXR, HDR_FORMAT_PNG])
        video_converter = LumaFluxSdrVideoToHdr.define_schema()
        video_saver = SaveHdr10Video.define_schema()
        self.assertEqual(video_converter.inputs[1].io_type, "VIDEO")
        self.assertEqual(video_converter.outputs[0].io_type, "HDR_VIDEO_STREAM")
        self.assertNotIn("SAI_", video_converter.outputs[0].io_type)
        self.assertEqual(video_saver.inputs[0].io_type, "HDR_VIDEO_STREAM")
        self.assertEqual(video_saver.category, "Sai/Saver/HDR")
        self.assertTrue(video_converter.display_name.endswith("Ψ"))
        self.assertTrue(video_saver.display_name.endswith("Ψ"))
        video_inputs = {value.id: value for value in video_converter.inputs}
        self.assertEqual(video_inputs["resize_mode"].default, "crop")
        self.assertTrue(video_inputs["shared_noise"].default)
        self.assertIsNone(getattr(video_inputs["shared_noise"], "label_on", None))
        self.assertIsNone(getattr(video_inputs["shared_noise"], "label_off", None))
        saver_inputs = {value.id: value for value in video_saver.inputs}
        self.assertEqual(
            saver_inputs["format"].options,
            [HDR_VIDEO_FORMAT, HDR_EXR_SEQUENCE_FORMAT, HDR_PNG_SEQUENCE_FORMAT],
        )


if __name__ == "__main__":
    unittest.main()
