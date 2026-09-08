# Load MyTM Re-ageing Model Ψ

Loads a MyTimeMachine/SAM checkpoint from `models/mytimemachine/`.
Known pSp, StyleGAN, IR-SE50, and DEX auxiliary checkpoints are intentionally
excluded from the dropdown because they are not complete inference checkpoints.

## Inputs

- `base_checkpoint`: general aging model. It defaults to
  `sam_ffhq_aging.pt`, which is also the global aging prior used during
  personalized inference.
- `personalized_checkpoint`: leave at `none` for general SAM aging, or select
  an identity-specific checkpoint such as Al Pacino's `iteration_10000.pt` to
  enable its latent blender.

The internal input IDs remain `checkpoint` and `global_checkpoint` so existing
workflows keep loading, but the visible labels and execution semantics follow
the base/personalized model roles above. Workflows saved with the former
personalized/global ordering are detected and migrated at execution time.

## Output

- `mytimemachine_model`: model managed by ComfyUI for the transform node.

The original research code is MIT licensed. Its required inference architecture
is vendored under `vendor/mytimemachine/` with attribution and license files.
