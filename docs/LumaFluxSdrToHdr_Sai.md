# LumaFlux SDR to HDR Ψ

Converts an ordinary BT.709 SDR `IMAGE` into a typed `HDR_IMAGE` containing
PQ-coded BT.2020 RGB values mastered at 1000 nits.

- `steps`: defaults to the upstream 8-step Euler transport.
- `resize_mode` uses the same geometry policy as the video node:
  - `none` requires exact multiples of 16;
  - `crop` (default) edge-pads the right/bottom for inference and crops the HDR
    result back to the original size without removing input content;
  - `resize` uses antialiased bicubic resampling to the nearest multiples of 16.
  The node always reports whether and how geometry was processed.
- `seed`: controls the training-matched bridge noise.
- `shared_noise`: for an IMAGE batch, enables the complete temporal-consistency
  bundle: shared bridge noise, previous applied RQS spline state, and EMA.
  Disable it to process batch items independently. It has no effect on a single
  image.
- `spline_ema`: previous-frame weight, default `0.8`; used only while
  `shared_noise` is enabled.
- `time_shift_factor`: 1.0 keeps the timestep grid uniform.
- `bridge_noise`: upstream default is 0.05.
- `tone_strength`: 1.0 applies the learned RQS tone field as trained.

The output is not directly interchangeable with a normal ComfyUI `IMAGE`,
because doing so would discard the transfer, primaries, mastering peak, and
range contract.
