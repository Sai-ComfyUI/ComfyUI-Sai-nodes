# MyTM Re-ageing Ψ

Transforms an aligned face to a target age using MyTimeMachine.

## Inputs

- `mytimemachine_model`: output from **Load MyTM Re-ageing Model Ψ**.
- `image`: aligned, tightly cropped FFHQ-style face image. The node resizes the
  model input to 256×256 internally; it does not perform landmark alignment.
- `target_age`: integer from 0 through 100, used when `age_mode` is `single`.
- `age_mode`: `single` for one image per input, or `range` for an age sequence.
- `range_start_age` / `range_end_age`: inclusive endpoints for range mode. Reverse
  ranges are supported.
- `range_step`: positive age interval. If the interval does not land exactly on
  the end age, the end age is still included.

## Output

- `aged_image`: 1024×1024 ComfyUI `IMAGE`. In range mode, results are ordered by
  age; for an input batch, every input image at one age is followed by every
  input image at the next age. For example, one image and ages 10 through 30
  produce a 21-frame IMAGE batch suitable for video nodes.

For best identity and framing, align the face first with **YuNet FFHQ Face
Align Ψ** or another FFHQ-compatible alignment node.
