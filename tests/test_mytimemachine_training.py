"""Tests for MyTimeMachine personalized training orchestration."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

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

from comfyui_sai_nodes import SaiNodesExtension
from comfyui_sai_nodes.nodes.image.mytimemachine_training import (
    MyTimeMachineTrainingStatusSai,
    StartMyTimeMachineTrainingSai,
    _launch_signature,
    _wait_for_training,
    resolve_training_output,
    validate_training_dataset,
)


class MyTimeMachineTrainingTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (32, 32), "black").save(dataset / "30_001.png")
        Image.new("RGB", (32, 32), "white").save(dataset / "70_002.jpg")
        return dataset

    def test_dataset_contract_reads_age_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._make_dataset(Path(temporary))
            resolved, images, ages = validate_training_dataset(str(dataset))
        self.assertEqual(resolved, dataset.resolve())
        self.assertEqual(len(images), 2)
        self.assertEqual(ages, {30, 70})

    def test_dataset_rejects_missing_age_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            Image.new("RGB", (16, 16)).save(dataset / "portrait.png")
            with self.assertRaisesRegex(ValueError, "filename must begin with an age"):
                validate_training_dataset(str(dataset))

    def test_output_filename_cannot_escape_selected_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "without directory components"):
            resolve_training_output("models", "../escaped.pt")
        with self.assertRaisesRegex(ValueError, "must end with .pt"):
            resolve_training_output("models", "model.ckpt")

    def test_start_fingerprint_is_stable_until_dataset_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._make_dataset(Path(temporary))
            values = {"dataset_directory": str(dataset), "max_steps": 100}
            first = _launch_signature(values)
            self.assertEqual(first, _launch_signature(values))
            Image.new("RGB", (32, 32), "red").save(dataset / "50_003.png")
            self.assertNotEqual(first, _launch_signature(values))

    def test_queue_interrupt_cancels_training_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            state_path.write_text(
                json.dumps({"status": "training", "progress": 3}), encoding="utf-8"
            )
            process = Mock(pid=4321)
            process.poll.return_value = None
            log_thread = Mock()
            from comfy import model_management

            with (
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.model_management.throw_exception_if_processing_interrupted",
                    side_effect=model_management.InterruptProcessingException(),
                ),
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training._terminate_process_tree"
                ) as terminate,
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.comfy.utils.ProgressBar"
                ),
            ):
                with self.assertRaises(model_management.InterruptProcessingException):
                    _wait_for_training(process, state_path, 100, "test-job", log_thread)
            terminate.assert_called_once_with(process)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "cancelled")

    def test_training_nodes_are_registered(self) -> None:
        nodes = asyncio.run(SaiNodesExtension().get_node_list())
        self.assertIn(StartMyTimeMachineTrainingSai, nodes)
        self.assertIn(MyTimeMachineTrainingStatusSai, nodes)
        schema = StartMyTimeMachineTrainingSai.define_schema()
        self.assertEqual(schema.node_id, "StartMyTimeMachineTraining_Sai")
        self.assertTrue(schema.is_output_node)
        self.assertEqual(schema.display_name, "Start MyTM Personalized Training Ψ")
        self.assertEqual(schema.category, "Sai/Re-ageing/MyTimeMachine/Training")
        status_schema = MyTimeMachineTrainingStatusSai.define_schema()
        self.assertEqual(status_schema.display_name, "MyTM Training Status Ψ")
        self.assertEqual(
            status_schema.category,
            "Sai/Re-ageing/MyTimeMachine/Training",
        )

    def test_start_writes_job_config_and_launches_background_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._make_dataset(root)
            jobs = root / "jobs"
            output = root / "models"
            fake_process = Mock(pid=4321)
            with (
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.TRAINING_JOBS_DIRECTORY",
                    jobs,
                ),
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.folder_paths.get_full_path_or_raise",
                    return_value=str(PACKAGE_ROOT / "models/mytimemachine/sam_ffhq_aging.pt"),
                ),
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.subprocess.Popen",
                    return_value=fake_process,
                ) as popen,
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.model_management.unload_all_models"
                ),
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.model_management.soft_empty_cache"
                ),
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training.threading.Thread"
                ),
                patch(
                    "comfyui_sai_nodes.nodes.image.mytimemachine_training._wait_for_training",
                    return_value={
                        "status": "completed",
                        "output_path": str((output / "subject.pt").resolve()),
                    },
                ),
            ):
                result = StartMyTimeMachineTrainingSai.execute(
                    str(dataset),
                    str(output),
                    "subject.pt",
                    "sam_ffhq_aging.pt",
                    100,
                    1,
                    0.0001,
                    7.0,
                    5.0,
                    0.1,
                    1.0,
                    0.005,
                    50,
                    1,
                    0,
                    False,
                )
            job_id, status, output_path, log_path = result.result
            self.assertEqual(status, "completed")
            self.assertEqual(output_path, str((output / "subject.pt").resolve()))
            self.assertTrue(Path(log_path).is_file())
            state = json.loads((jobs / job_id / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["pid"], 4321)
            config = json.loads((jobs / job_id / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["checkpoint_interval"], 50)
            self.assertEqual(config["adaptive_w_norm_lambda"], 7.0)
            popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
