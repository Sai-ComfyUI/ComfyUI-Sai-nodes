"""Background personalized-training nodes for MyTimeMachine."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui

from .mytimemachine import (
    MODEL_CATEGORY,
    MODEL_DIRECTORY,
    get_base_checkpoint_options,
    get_default_base_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONALIZED_MODEL_DIRECTORY = MODEL_DIRECTORY / "personalized"
TRAINING_JOBS_DIRECTORY = PROJECT_ROOT / ".dev" / "mytimemachine_training"
TRAINING_RUNNER = PROJECT_ROOT / "vendor" / "mytimemachine" / "train_runner.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tiff"}
AGE_PREFIX = re.compile(r"^(\d{1,3})(?:[_. ]|$)")
ACTIVE_JOB_STATUSES = {"starting", "running", "initializing", "training"}


def validate_training_dataset(dataset_directory: str) -> tuple[Path, list[Path], set[int]]:
    dataset = Path(dataset_directory).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    images = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"Dataset contains no supported images: {dataset}")
    ages: set[int] = set()
    invalid: list[str] = []
    for image in images:
        match = AGE_PREFIX.match(image.name)
        if match is None or not 0 <= int(match.group(1)) <= 100:
            invalid.append(image.name)
        else:
            ages.add(int(match.group(1)))
    if invalid:
        sample = ", ".join(invalid[:5])
        raise ValueError(
            "Every dataset filename must begin with an age from 0 to 100 "
            f"(for example 35_001.png). Invalid: {sample}"
        )
    if len(ages) < 2:
        raise ValueError("Personalized training requires images from at least two ages.")
    return dataset, images, ages


def resolve_training_output(output_directory: str, output_filename: str) -> Path:
    directory = Path(output_directory).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory = directory.resolve()
    filename = output_filename.strip()
    if not filename or Path(filename).name != filename:
        raise ValueError("output_filename must be a filename without directory components.")
    if Path(filename).suffix.lower() != ".pt":
        raise ValueError("output_filename must end with .pt.")
    return directory / filename


def _is_process_alive(pid: object) -> bool:
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return False
    if process_id <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(process_id)
    except ImportError:
        try:
            os.kill(process_id, 0)
        except (OSError, ValueError):
            return False
        return True


def _refresh_stale_state(state_path: Path, state: dict) -> dict:
    if state.get("status") in ACTIVE_JOB_STATUSES and state.get("pid"):
        if not _is_process_alive(state["pid"]):
            state["status"] = "interrupted"
            state["error"] = "Training process is no longer running."
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return state


def _dataset_signature(dataset_directory: object) -> list[tuple[str, int, int]]:
    try:
        dataset = Path(str(dataset_directory)).expanduser().resolve()
        if not dataset.is_dir():
            return []
        return [
            (str(path.relative_to(dataset)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(dataset.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    except OSError:
        return []


def _launch_signature(values: dict) -> str:
    payload = {key: value for key, value in values.items() if key != "dataset_files"}
    payload["dataset_files"] = values.get(
        "dataset_files", _dataset_signature(values.get("dataset_directory", ""))
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _find_active_output_job(output_path: Path) -> dict | None:
    if not TRAINING_JOBS_DIRECTORY.is_dir():
        return None
    for state_path in TRAINING_JOBS_DIRECTORY.glob("*/state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state = _refresh_stale_state(state_path, state)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            state.get("status") in ACTIVE_JOB_STATUSES
            and Path(state.get("output_path", "")).resolve() == output_path
        ):
            return state
    return None


def _mirror_training_log(
    process: subprocess.Popen, log_path: Path, job_id: str, state_path: Path
) -> None:
    prefix = f"[MyTimeMachine Training {job_id}]"
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                message = line.rstrip()
                if message:
                    print(f"{prefix} {message}", flush=True)
        exit_code = process.wait()
        print(f"{prefix} process exited with code {exit_code}", flush=True)
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") in ACTIVE_JOB_STATUSES:
                state["status"] = "interrupted"
                state["error"] = f"Training process exited before completion (code {exit_code})."
                state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except (OSError, json.JSONDecodeError):
            pass
    except BaseException as exc:
        print(f"{prefix} log monitor failed: {type(exc).__name__}: {exc}", flush=True)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate the trainer and any DataLoader children created by it."""

    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([*children, parent], timeout=5)
            for remaining in alive:
                remaining.kill()
            psutil.wait_procs(alive, timeout=5)
            return
        except psutil.Error:
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _write_job_state(state_path: Path, **values) -> dict:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    state.update(values)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(state_path)
    return state


def _wait_for_training(
    process: subprocess.Popen,
    state_path: Path,
    max_steps: int,
    job_id: str,
    log_thread: threading.Thread,
) -> dict:
    """Keep the ComfyUI job active, update its progress, and honor Interrupt."""

    progress = comfy.utils.ProgressBar(max_steps)
    last_progress = -1
    try:
        while process.poll() is None:
            model_management.throw_exception_if_processing_interrupted()
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                current = int(state.get("progress", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                current = last_progress
            if current >= 0 and current != last_progress:
                progress.update_absolute(current, max_steps)
                last_progress = current
            time.sleep(0.25)
    except model_management.InterruptProcessingException:
        print(f"[MyTimeMachine Training {job_id}] cancellation requested", flush=True)
        _terminate_process_tree(process)
        log_thread.join(timeout=5)
        _write_job_state(
            state_path,
            status="cancelled",
            error="Training was cancelled from the ComfyUI queue.",
        )
        print(f"[MyTimeMachine Training {job_id}] cancelled", flush=True)
        raise

    exit_code = process.wait()
    log_thread.join(timeout=5)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Training ended without a readable state: {exc}") from exc
    if exit_code != 0 or state.get("status") != "completed":
        error = state.get("error", f"trainer exited with code {exit_code}")
        raise RuntimeError(f"MyTimeMachine training failed: {error}")
    progress.update_absolute(max_steps, max_steps)
    return state


def _read_state(job_id: str) -> dict:
    if job_id == "latest":
        latest = TRAINING_JOBS_DIRECTORY / "latest.txt"
        if not latest.is_file():
            raise FileNotFoundError("No MyTimeMachine training job has been started.")
        job_id = latest.read_text(encoding="utf-8").strip()
    state_path = TRAINING_JOBS_DIRECTORY / job_id / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Unknown MyTimeMachine training job: {job_id}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return _refresh_stale_state(state_path, state)


class StartMyTimeMachineTrainingSai(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        bases = get_base_checkpoint_options()
        return io.Schema(
            node_id="StartMyTimeMachineTraining_Sai",
            display_name="Start MyTM Personalized Training Ψ",
            category="Sai/Re-ageing/MyTimeMachine/Training",
            description=(
                "Runs official-style MyTimeMachine personalized training as a native "
                "ComfyUI queue job with progress reporting and cancellation support."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "dataset_directory",
                    default="",
                    tooltip="Folder of aligned faces named AGE_index.ext, e.g. 35_001.png.",
                ),
                io.String.Input(
                    "output_directory",
                    default=str(PERSONALIZED_MODEL_DIRECTORY),
                ),
                io.String.Input("output_filename", default="personalized_model.pt"),
                io.Combo.Input(
                    "base_checkpoint",
                    options=bases,
                    default=get_default_base_checkpoint(bases),
                ),
                io.Int.Input("max_steps", default=10000, min=1, max=500000, step=100),
                io.Int.Input("batch_size", default=1, min=1, max=32, step=1),
                io.Float.Input(
                    "learning_rate",
                    default=0.0001,
                    min=0.000001,
                    max=0.01,
                    step=0.000001,
                    advanced=True,
                ),
                io.Float.Input(
                    "adaptive_w_norm_lambda",
                    default=7.0,
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "Personalization strength from Eqn. 7. Start at 7; higher "
                        "values (10-30) can reduce underfitting, while lower values "
                        "such as 5 can reduce overfitting."
                    ),
                ),
                io.Int.Input("workers", default=2, min=0, max=16, step=1, advanced=True),
                io.Boolean.Input("overwrite_existing", default=False, advanced=True),
                io.Float.Input(
                    "aging_lambda", default=5.0, min=0.0, max=100.0, step=0.1,
                    advanced=True,
                    tooltip="Strength of matching the requested target age.",
                ),
                io.Float.Input(
                    "identity_lambda", default=0.1, min=0.0, max=10.0, step=0.01,
                    advanced=True,
                    tooltip="Identity-preservation loss weight.",
                ),
                io.Float.Input(
                    "cycle_lambda", default=1.0, min=0.0, max=20.0, step=0.1,
                    advanced=True,
                    tooltip="Cycle-consistency weight for reconstructing the source identity.",
                ),
                io.Float.Input(
                    "w_norm_lambda", default=0.005, min=0.0, max=1.0, step=0.001,
                    advanced=True,
                    tooltip="Base latent W-norm regularization weight.",
                ),
                io.Int.Input(
                    "checkpoint_interval", default=2000, min=1, max=500000, step=100,
                    advanced=True,
                    tooltip=(
                        "Save a recoverable checkpoint in the job folder every N steps. "
                        "Each checkpoint is approximately 2.3 GB."
                    ),
                ),
                io.Int.Input(
                    "progress_log_interval", default=1, min=1, max=10000, step=1,
                    advanced=True,
                    tooltip="Print progress, loss, elapsed time and ETA to the ComfyUI log every N steps.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="job_id"),
                io.String.Output(display_name="status"),
                io.String.Output(display_name="output_path"),
                io.String.Output(display_name="log_path"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _launch_signature(kwargs)

    @classmethod
    def execute(
        cls,
        dataset_directory: str,
        output_directory: str,
        output_filename: str,
        base_checkpoint: str,
        max_steps: int,
        batch_size: int,
        learning_rate: float,
        adaptive_w_norm_lambda: float,
        aging_lambda: float,
        identity_lambda: float,
        cycle_lambda: float,
        w_norm_lambda: float,
        checkpoint_interval: int,
        progress_log_interval: int,
        workers: int,
        overwrite_existing: bool,
    ) -> io.NodeOutput:
        dataset, images, ages = validate_training_dataset(dataset_directory)
        output_path = resolve_training_output(output_directory, output_filename)
        if output_path.exists() and not overwrite_existing:
            raise FileExistsError(
                f"Output already exists: {output_path}. Enable overwrite_existing to replace it."
            )
        launch_signature = _launch_signature(
            {
                "dataset_directory": str(dataset),
                "dataset_files": [
                    (str(path.relative_to(dataset)), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in images
                ],
                "output_path": str(output_path),
                "base_checkpoint": base_checkpoint,
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "adaptive_w_norm_lambda": float(adaptive_w_norm_lambda),
                "aging_lambda": float(aging_lambda),
                "identity_lambda": float(identity_lambda),
                "cycle_lambda": float(cycle_lambda),
                "w_norm_lambda": float(w_norm_lambda),
                "checkpoint_interval": min(int(checkpoint_interval), int(max_steps)),
                "progress_log_interval": int(progress_log_interval),
                "workers": int(workers),
                "overwrite_existing": bool(overwrite_existing),
            }
        )
        active_job = _find_active_output_job(output_path)
        if active_job is not None:
            if active_job.get("launch_signature") == launch_signature:
                summary = (
                    "Identical training job is already active; no duplicate was started.\n"
                    f"job_id: {active_job.get('job_id')}\n"
                    f"status: {active_job.get('status')}\n"
                    f"progress: {active_job.get('progress', 0)}/{active_job.get('max_steps', max_steps)}"
                )
                return io.NodeOutput(
                    active_job.get("job_id", ""),
                    active_job.get("status", "running"),
                    active_job.get("output_path", str(output_path)),
                    active_job.get("log_path", ""),
                    ui=ui.PreviewText(summary),
                )
            raise RuntimeError(
                "A MyTimeMachine training job is already using this output path: "
                f"{active_job.get('job_id')} (PID {active_job.get('pid')})."
            )
        base_path = Path(
            folder_paths.get_full_path_or_raise(MODEL_CATEGORY, base_checkpoint)
        ).resolve()
        required_models = {
            "base_checkpoint": base_path,
            "ir_se50": MODEL_DIRECTORY / "model_ir_se50.pth",
            "age_predictor": MODEL_DIRECTORY / "dex_age_classifier.pth",
            "lpips_alex": MODEL_DIRECTORY / "lpips_alex_v0.1.pth",
            "pretrained_psp": MODEL_DIRECTORY / "psp_ffhq_encode.pt",
            "stylegan_ffhq": MODEL_DIRECTORY / "stylegan2-ffhq-config-f.pt",
        }
        missing = [name for name, path in required_models.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing training model files: {', '.join(missing)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        TRAINING_JOBS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job_directory = TRAINING_JOBS_DIRECTORY / job_id
        job_directory.mkdir()
        config_path = job_directory / "config.json"
        state_path = job_directory / "state.json"
        log_path = job_directory / "training.log"
        config = {
            "job_id": job_id,
            "job_directory": str(job_directory),
            "state_path": str(state_path),
            "dataset_directory": str(dataset),
            "dataset_image_count": len(images),
            "dataset_ages": sorted(ages),
            "output_path": str(output_path),
            "overwrite_existing": bool(overwrite_existing),
            "base_checkpoint": str(base_path),
            "model_paths": {name: str(path.resolve()) for name, path in required_models.items()},
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "adaptive_w_norm_lambda": float(adaptive_w_norm_lambda),
            "aging_lambda": float(aging_lambda),
            "identity_lambda": float(identity_lambda),
            "cycle_lambda": float(cycle_lambda),
            "w_norm_lambda": float(w_norm_lambda),
            "checkpoint_interval": min(int(checkpoint_interval), int(max_steps)),
            "progress_log_interval": int(progress_log_interval),
            "workers": int(workers),
        }
        config["launch_signature"] = launch_signature
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        state = {
            "job_id": job_id,
            "status": "starting",
            "progress": 0,
            "max_steps": int(max_steps),
            "output_path": str(output_path),
            "log_path": str(log_path),
            "dataset_image_count": len(images),
            "dataset_ages": sorted(ages),
            "launch_signature": config["launch_signature"],
            "checkpoint_directory": str(job_directory / "checkpoints"),
            "checkpoint_interval": config["checkpoint_interval"],
            "progress_log_interval": config["progress_log_interval"],
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        (TRAINING_JOBS_DIRECTORY / "latest.txt").write_text(job_id, encoding="utf-8")

        model_management.unload_all_models()
        model_management.soft_empty_cache()
        log_path.touch()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(TRAINING_RUNNER), "--config", str(config_path)],
                cwd=str(TRAINING_RUNNER.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except BaseException:
            state["status"] = "failed_to_start"
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            raise
        current_state = json.loads(state_path.read_text(encoding="utf-8"))
        if current_state.get("status") == "starting":
            current_state["status"] = "running"
        current_state["pid"] = process.pid
        state_path.write_text(json.dumps(current_state, indent=2), encoding="utf-8")
        log_thread = threading.Thread(
            target=_mirror_training_log,
            args=(process, log_path, job_id, state_path),
            name=f"mytimemachine-log-{job_id}",
            daemon=True,
        )
        log_thread.start()
        print(
            f"[MyTimeMachine Training {job_id}] started PID {process.pid}; "
            f"dataset={len(images)} images; steps={max_steps}; output={output_path}",
            flush=True,
        )
        completed_state = _wait_for_training(
            process, state_path, int(max_steps), job_id, log_thread
        )
        summary = (
            f"Training completed\njob_id: {job_id}\n"
            f"output: {output_path}\nlog: {log_path}"
        )
        return io.NodeOutput(
            job_id,
            completed_state.get("status", "completed"),
            completed_state.get("output_path", str(output_path)),
            completed_state.get("log_path", str(log_path)),
            ui=ui.PreviewText(summary),
        )


class MyTimeMachineTrainingStatusSai(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyTimeMachineTrainingStatus_Sai",
            display_name="MyTM Training Status Ψ",
            category="Sai/Re-ageing/MyTimeMachine/Training",
            is_output_node=True,
            not_idempotent=True,
            inputs=[io.String.Input("job_id", default="latest")],
            outputs=[
                io.String.Output(display_name="status"),
                io.String.Output(display_name="details_json"),
                io.String.Output(display_name="output_path"),
                io.String.Output(display_name="log_path"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time_ns()

    @classmethod
    def execute(cls, job_id: str) -> io.NodeOutput:
        state = _read_state(job_id.strip() or "latest")
        details = json.dumps(state, ensure_ascii=False, indent=2)
        return io.NodeOutput(
            state.get("status", "unknown"),
            details,
            state.get("output_path", ""),
            state.get("log_path", ""),
            ui=ui.PreviewText(details),
        )


__all__ = [
    "MyTimeMachineTrainingStatusSai",
    "StartMyTimeMachineTrainingSai",
    "resolve_training_output",
    "validate_training_dataset",
]
