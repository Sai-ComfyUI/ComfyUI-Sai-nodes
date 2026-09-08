# Node usage

All nodes are registered under the `Sai` category. Search for the `Ψ` suffix to
find this package's public nodes.

## Conditioning

### Multi Reference Latent Ψ

Scales and VAE-encodes multiple reference images, then appends the resulting
latents to both positive and negative conditioning for supported edit models.
It also returns the scaled first image and the original VAE. See
[the node reference](Flux2KleinMultiReferenceLatent_Sai.md).

## Latent upscaling

1. Add **Load LUA FLUX Model Ψ** and choose the official download entry or an
   existing `lua_flux.pth`.
2. Connect its model output and a sampled FLUX.1 latent to
   **LUA FLUX Latent Upscale Ψ**.
3. Select x2 or x4 and decode the output with the matching FLUX.1 VAE.

The model is for FLUX.1's 16-channel latent format and intentionally rejects
FLUX.2 latents. See [loader](LoadLuaFluxModel_Sai.md) and
[upscaler](LuaFluxLatentUpscale_Sai.md) references.

## MyTimeMachine re-ageing

The recommended portrait workflow is:

```text
IMAGE -> YuNet FFHQ Face Align Ψ -> MyTM Re-ageing Ψ -> YuNet FFHQ Face Restore Ψ
                       context --------------------------^
```

1. Install the files for the features you need from [Model downloads](MODELS.md).
2. Use **Load MyTM Re-ageing Model Ψ** with `sam_ffhq_aging.pt` for general
   re-ageing. Select a personalized checkpoint only when one is available.
3. Align the source portrait with **YuNet FFHQ Face Align Ψ**.
4. Connect the aligned face and loaded model to **MyTM Re-ageing Ψ**. Single-age
   and inclusive age-range output are supported.
5. Connect the generated image and saved alignment context to
   **YuNet FFHQ Face Restore Ψ** to composite it into the original image.

Detailed references:

- [Load MyTimeMachine model](LoadMyTimeMachineModel_Sai.md)
- [Face align](MyTimeMachineFaceAlign_Sai.md)
- [Re-ageing](MyTimeMachineAgeTransform_Sai.md)
- [Face restore](MyTimeMachineFaceRestore_Sai.md)

## MyTimeMachine personalized training

**Start MyTM Personalized Training Ψ** expects recursively scanned FFHQ-aligned
images whose filenames begin with the ground-truth age, such as `35_001.png`.
At least two distinct ages are required. Training stays in ComfyUI's running
queue and can be cancelled there. Use **MyTM Training Status Ψ** to review a
previous job or its logs.

Training requires substantially more model files than inference. Install the
complete training set in [Model downloads](MODELS.md), then see the
[training](StartMyTimeMachineTraining_Sai.md) and
[status](MyTimeMachineTrainingStatus_Sai.md) references.

## Labeled image collage

Connect images to the growing image sockets, enter one label per line, then
choose horizontal or vertical layout, cell sizing, wrapping, label position,
font, and colors. Named presets are saved in browser local storage for the
current ComfyUI address; image connections are not stored in presets.

See [Labeled Image Collage](LabeledImageCollage_Sai.md).

## Restart action

The frontend restart action asks the ComfyUI process to exit with code `75`.
It restarts automatically only when the process is launched by a supervisor or
script that recognizes this code. Otherwise, start ComfyUI again manually.
