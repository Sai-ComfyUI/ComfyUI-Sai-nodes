# Save HDR Image Ψ

Saves a typed `HDR_IMAGE` without lossy compression or chroma subsampling.

- `OpenEXR (32-bit float, linear BT.2020)` is the default master format. PQ is
  decoded to linear BT.2020 RGB and normalized so `1.0` equals the mastering
  peak recorded in the accompanying JSON sidecar.
- `PNG (16-bit, PQ BT.2020)` losslessly stores PQ code values and matches the
  still-frame format used by the upstream LumaFlux inference pipeline. It also
  embeds PNG `cICP` values 9/16/0/1 for BT.2020 primaries, PQ transfer, RGB
  matrix, and full range.

The node validates the HDR contract before saving and is intentionally placed
under `Sai/Saver/HDR`, so it can be reused by future SDR-to-HDR engines. AVIF
is not offered as a master because its 10-bit YUV420 path is lossy and
subsamples chroma. HDR10 video still needs a separate writer with
mastering-display and content-light metadata.
