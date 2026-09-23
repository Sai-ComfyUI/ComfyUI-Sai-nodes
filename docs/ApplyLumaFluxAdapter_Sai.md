# Apply LumaFlux Adapter Ψ

Combines an already loaded native ComfyUI FLUX.1-dev `MODEL`, Flux `VAE`,
SigLIP `CLIP_VISION`, and a LumaFlux adapter checkpoint into a typed
`LUMAFLUX_MODEL` runtime.

The node accepts NVFP4, FP8, or BF16 FLUX.1-dev checkpoints without converting
their weights to Diffusers. FLUX.1 Schnell is rejected. SigLIP must be the
SO400M patch14-384 vision model loaded from `models/clip_vision`.

The adapter selector scans both `ComfyUI/models/lumaflux/` and this package's
`models/lumaflux/` directory.
