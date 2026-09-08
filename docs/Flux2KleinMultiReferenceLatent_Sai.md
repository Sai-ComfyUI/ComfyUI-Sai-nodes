# Multi Reference Latent Ψ

Scales and VAE-encodes an ordered set of reference images, then appends the
latent list to both positive and negative conditioning using ComfyUI's standard
`reference_latents` metadata. Compatibility and result quality depend on the
edit model receiving the conditioning.
## Inputs

- `positive_conditioning` / `negative_conditioning`: text conditioning that
  receives the same ordered reference list.
- `vae`: an edit-model-compatible VAE used to encode every reference.
- `upscale_method`: resize filter applied before encoding.
- `megapixels`: target total pixel count for each image; aspect ratio is kept.
- `resolution_steps`: advanced dimension-rounding multiple.
- `image_1` through the growing image sockets: references in model input order,
  up to 16 images.

## Outputs

- `positive_conditioning` / `negative_conditioning`: copied conditioning with
  the encoded reference latents appended.
- `first_image_scaled`: the resized first reference for preview or reuse.
- `vae`: the original VAE for downstream nodes.

This node was initially developed for Flux.2 Klein image editing but uses the
generic ComfyUI reference-latent convention rather than a model-specific data
type.
