# LUA FLUX Latent Upscale Ψ

Runs the official LUA x2 or x4 head on a sampled FLUX.1 VAE latent before image
decoding.

## Inputs

- `lua_model`: Output from **Load LUA FLUX Model Ψ**.
- `latent`: A standard ComfyUI FLUX.1 latent shaped `[B, 16, H, W]`.
- `scale`: `x2` or `x4` spatial latent upscaling.

## Output

- `latent_upscaled`: Connect directly to **VAE Decode** with the matching
  FLUX.1 VAE.

## Compatibility

This checkpoint is for FLUX.1's 16-channel latent space. FLUX.2 uses a
128-channel latent format and is rejected with a clear error. Existing latent
metadata is preserved, except `noise_mask`, which is removed because its source
resolution would no longer match the upscaled samples.
