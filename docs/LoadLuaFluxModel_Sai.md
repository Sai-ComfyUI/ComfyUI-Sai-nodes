# Load LUA FLUX Model Ψ

Loads the official [vaskers5/LUA-FLUX](https://huggingface.co/vaskers5/LUA-FLUX)
checkpoint as a ComfyUI latent-upscale model.

## Inputs

- `model_name`: Choose the official download action, or a local checkpoint in
  `ComfyUI/models/latent_upscale_models`.
- `precision`: `fp32` matches the official recommendation. `bf16` reduces model
  memory at a possible quality cost.

## Output

- `lua_model`: Connect to **LUA FLUX Latent Upscale Ψ**.

The download action stores `lua_flux.pth` in ComfyUI's standard
`models/latent_upscale_models` directory. Subsequent runs reuse that file.
