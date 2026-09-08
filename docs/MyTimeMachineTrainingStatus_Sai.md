# MyTM Training Status Ψ

Reads a background personalized-training job state.

- `job_id`: use the ID returned by the start node, or leave `latest` to inspect
  the most recently started job.
- `status`: `starting`, `running`, `initializing`, `training`, `completed`,
  `failed`, `cancelled` or `interrupted`.
- `details_json`: progress, percentage, elapsed time, ETA, maximum steps,
  latest loss metrics, PID, checkpoint directory and any captured traceback.
- `output_path`: final requested checkpoint path.
- `log_path`: full training log for diagnosis.

Run this output node again to refresh the current state.

For new native-queue training runs, this node is mainly useful for reviewing
history after completion. Current progress and cancellation are available
directly in ComfyUI's queue UI.
