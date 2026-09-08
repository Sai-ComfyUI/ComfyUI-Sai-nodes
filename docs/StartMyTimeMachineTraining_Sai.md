# Start MyTM Personalized Training Ψ

Runs personalized MyTimeMachine training as a native ComfyUI queue job. It
remains in the Running section until training completes, fails, or is cancelled.
Pending jobs can be removed normally; cancelling the running job terminates the
trainer and its DataLoader child processes and records the state as `cancelled`.

## Dataset

`dataset_directory` must contain FFHQ-aligned face images. Every filename must
begin with the ground-truth age from 0 through 100:

```text
subject_faces/
├── 30_001.png
├── 35_002.jpg
├── 60_003.png
└── 70_004.png
```

At least two different ages are required. Subdirectories are scanned
recursively. This node validates naming but does not automatically align a
whole dataset.

## Output

- `output_directory` defaults to `models/mytimemachine/personalized`.
- `output_filename` defaults to `personalized_model.pt` and must be a plain
  `.pt` filename.
- Existing output is protected unless `overwrite_existing` is enabled.
- An active job locks its output path. Re-queueing identical inputs reuses the
  cached result or active job instead of launching another trainer.
- The completed model appears in the personalized checkpoint menu after model
  definitions are refreshed.

## Main settings

- `base_checkpoint`: normally `sam_ffhq_aging.pt`.
- `max_steps`: defaults to 10,000.
- `batch_size`: defaults to 1 for safer VRAM use on a 16 GB GPU.
- `adaptive_w_norm_lambda`: official recommendation is 7. Higher values keep
  can help when a result is underfit and too close to global aging (the authors
  suggest 10–30); lower values such as 5 can reduce overfitting.

Advanced loss controls preserve the official defaults:

- `aging_lambda`: target-age supervision strength, default 5.
- `identity_lambda`: identity-preservation strength, default 0.1.
- `cycle_lambda`: source reconstruction consistency, default 1.
- `w_norm_lambda`: base latent regularization, default 0.005.
- `checkpoint_interval`: writes recoverable `iteration_N.pt` files inside the
  job's checkpoint directory, default every 2,000 steps as in the official
  command. Each checkpoint is approximately 2.3 GB.
- `progress_log_interval`: prints progress, loss, elapsed time and ETA into the
  ComfyUI log, default every step. Full loss metrics remain every 50 steps.

Changing these loss weights can destabilize personalization. Adjust
`adaptive_w_norm_lambda` first and leave the other official defaults unchanged
unless a controlled comparison indicates a specific problem.

## Progress and caching

The trainer itself remains an isolated subprocess, while the node stays active
in the ComfyUI queue and forwards progress to ComfyUI's native progress bar.
The server log also receives lines such as:

```text
[MyTimeMachine Training JOB_ID] Progress 250/10000 (2.5%) | loss=0.42100 | elapsed=3200s | ETA=124800s
```

The node fingerprint includes all inputs and image filenames, sizes and
modification times in the dataset. Identical inputs therefore use ComfyUI's
normal cache. Changing a parameter or adding/replacing a dataset image starts a
new run. A second server-side guard prevents different workflows or a restarted
ComfyUI instance from launching concurrent jobs to the same output path.

Cancelled or failed executions are not stored as successful cache entries, so
the same settings can be retried after resolving the problem.

The output sockets provide the job ID, initial status, expected checkpoint path
and log path. Connect or copy the job ID into **MyTM Training Status
Ψ** to inspect progress later. The status JSON also includes percent, elapsed
seconds, ETA, latest metrics and the intermediate checkpoint directory.
