# LumaFlux SDR Video to HDR Ψ

Accepts ComfyUI's native `VIDEO` value and creates a deferred
`HDR_VIDEO_STREAM`. It does not turn the video into an `IMAGE` batch.

- Frames are decoded from `VIDEO.get_stream_source()` only when a downstream
  saver consumes the stream.
- `shared_noise` defaults on. It enables the complete temporal-consistency
  bundle: the first frame's bridge noise is reused, the previous applied RQS
  spline state is carried forward, and `spline_ema` is applied. Disable it to
  process every frame independently.
- `spline_ema` defaults to the upstream video value `0.8` and is ignored while
  `shared_noise` is off.
- `resize_mode` matches the image node: `none` requires exact multiples of 16,
  `crop` edge-pads for inference then restores the source size, and `resize`
  resamples to the nearest multiples of 16. Processing is reported on the node.
- The stream preserves source frame rate and the active ComfyUI trim window.

Connect **Load Video** or **Trim Video** directly. Cropped or rotated VIDEO
wrappers are currently rejected when their declared dimensions differ from
the underlying stream, preventing an unnoticed geometry mismatch.
