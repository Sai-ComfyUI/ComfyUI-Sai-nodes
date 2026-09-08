"""Isolated background runner for personalized MyTimeMachine training."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import shutil
import sys
import traceback
import warnings


# PyTorch currently imports the deprecated ``pynvml`` compatibility package
# when another dependency installed it. This warning is external to the
# training job and has no effect on CUDA/NVML operation, so keep the training
# log focused while leaving every other warning visible.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
    module=r"torch\.cuda\.__init__",
)


VENDOR_ROOT = Path(__file__).resolve().parent
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


def update_state(state_path: Path, **values) -> None:
    state = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(values)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(state_path)


def make_options(config: dict) -> Namespace:
    models = config["model_paths"]
    job_directory = Path(config["job_directory"])
    return Namespace(
        gpus=1,
        train_dataset=config["dataset_directory"],
        exp_dir=str(job_directory),
        debug_dir=str(job_directory / "debug"),
        state_path=config["state_path"],
        dataset_type="ffhq_aging",
        input_nc=4,
        label_nc=0,
        output_size=1024,
        batch_size=config["batch_size"],
        test_batch_size=1,
        workers=config["workers"],
        test_workers=0,
        learning_rate=config["learning_rate"],
        optim_name="ranger",
        train_encoder=True,
        train_decoder=False,
        start_from_latent_avg=False,
        start_from_encoded_w_plus=True,
        lpips_lambda=0.1,
        id_lambda=config["identity_lambda"],
        l2_lambda=0.25,
        w_norm_lambda=config["w_norm_lambda"],
        aging_lambda=config["aging_lambda"],
        cycle_lambda=config["cycle_lambda"],
        lpips_lambda_crop=0.6,
        l2_lambda_crop=1.0,
        lpips_lambda_aging=0.1,
        l2_lambda_aging=0.25,
        stylegan_weights=models["stylegan_ffhq"],
        checkpoint_path=config["base_checkpoint"],
        max_steps=config["max_steps"],
        image_interval=100,
        board_interval=50,
        progress_log_interval=config["progress_log_interval"],
        val_interval=1000,
        save_interval=config["checkpoint_interval"],
        target_age="uniform_random",
        use_weighted_id_loss=True,
        pretrained_psp_path=models["pretrained_psp"],
        adaptive_w_norm_lambda=config["adaptive_w_norm_lambda"],
        nearest_neighbor_id_loss_lambda=1.0,
    )


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_path = Path(config["state_path"])
    try:
        update_state(state_path, status="initializing", progress=0)
        from configs.paths_config import model_paths

        model_paths.update(
            {
                "ir_se50": config["model_paths"]["ir_se50"],
                "age_predictor": config["model_paths"]["age_predictor"],
                "lpips_alex": config["model_paths"]["lpips_alex"],
                "pretrained_psp": config["model_paths"]["pretrained_psp"],
                "stylegan_ffhq": config["model_paths"]["stylegan_ffhq"],
            }
        )
        from configs import data_configs
        from training.coach_aging_delta import Coach

        dataset = config["dataset_directory"]
        data_configs.DATASETS["ffhq_aging"].update(
            {
                "train_source_root": dataset,
                "train_target_root": dataset,
                "test_source_root": dataset,
                "test_target_root": dataset,
            }
        )
        options = make_options(config)
        Path(options.exp_dir).mkdir(parents=True, exist_ok=True)
        (Path(options.exp_dir) / "options.json").write_text(
            json.dumps(vars(options), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Training initialized: {config['dataset_image_count']} images, "
            f"{config['max_steps']} steps, batch size {config['batch_size']}",
            flush=True,
        )
        update_state(state_path, status="training", progress=0)
        coach = Coach(options)
        coach.train()

        generated = Path(options.exp_dir) / "checkpoints" / f"iteration_{options.max_steps}.pt"
        if not generated.is_file():
            raise FileNotFoundError(f"Expected final checkpoint was not created: {generated}")
        output = Path(config["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if not config["overwrite_existing"]:
                raise FileExistsError(f"Output already exists: {output}")
            output.unlink()
        shutil.move(str(generated), str(output))
        update_state(
            state_path,
            status="completed",
            progress=options.max_steps,
            output_path=str(output),
        )
        print(f"Training completed: {output}", flush=True)
    except BaseException as exc:
        update_state(
            state_path,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
