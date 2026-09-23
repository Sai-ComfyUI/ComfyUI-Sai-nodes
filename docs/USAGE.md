# Node usage

All nodes are registered under the `Sai` category. Search for the `Ψ` suffix to
find this package's public nodes.

## Moonland bridge

Pair the unpacked Moonland Bridge extension from **Sai → Moonland Bridge** and
keep both the target Moonland page and ComfyUI open in the same browser.

**Moonland Resolve Tool Ψ** accepts an existing Lab topic or generation URL and
returns a versioned `MOONLAND_TOOL_CONTRACT`. Resolution is read-only and only
uses the exact signed-in Moonland tab for that URL. It does not submit a
generation, upload media, or expose browser credentials. See
[the resolver reference](MoonlandResolveTool_Sai.md).

The image resource nodes reuse or upload PNG, JPEG, and WebP content through the
same credential-free bridge. Video and audio transport are not yet implemented.

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

## LumaFlux SDR to HDR

```text
Load Diffusion Model ─ MODEL ─┐
VAE Loader ─────────── VAE ───┼─ Apply LumaFlux Adapter Ψ ─ LUMAFLUX_MODEL
CLIP Vision Loader ─ CLIP_VISION ┘                         │
SDR IMAGE ─────────────────────────────────────────────────┴─ LumaFlux SDR to HDR Ψ
                                                                 │ HDR_IMAGE
                                                                 ├─ HDR to SDR Preview Ψ
                                                                 └─ Save HDR Image Ψ

Load Video ─ VIDEO ────────────────────────────────────────┴─ LumaFlux SDR Video to HDR Ψ
                                                                 │ HDR_VIDEO_STREAM
                                                                 └─ Save HDR10 Video Ψ
```

Use FLUX.1-dev (NVFP4, FP8, or BF16), Flux `ae.safetensors`, SigLIP SO400M
patch14-384, and a released LumaFlux adapter. The source IMAGE is an ordinary
BT.709 SDR image; it does not need to be HDR. LumaFlux requires dimensions
divisible by 16. Both solver nodes expose the same `resize_mode`: `none` is
strict, `crop` (default) edge-pads and restores the source size, and `resize`
resamples to the nearest multiples of 16. The node displays every geometry
operation. Eight steps and bridge noise 0.05 reproduce the upstream inference
defaults.

`HDR_IMAGE` stores PQ code values plus BT.2020 and 1000-nit metadata. It is
intentionally distinct from an ordinary `IMAGE`, so metadata cannot be silently
lost. The preview node performs BT.2446 Method C tone mapping to BT.709 and is
not an HDR export path.

For still masters, **Save HDR Image Ψ** defaults to lossless float32 OpenEXR
containing normalized linear BT.2020 RGB (`1.0` equals the mastering peak).
It can alternatively write the upstream-compatible lossless 16-bit PNG that
stores PQ/BT.2020 code values and embeds a `cICP` HDR declaration. A JSON
sidecar records the transfer, primaries, white point, and mastering peak. Lossy
10-bit 4:2:0 AVIF is intentionally not used as the still-master format.

For video, connect ComfyUI's native **Load Video** output directly to
**LumaFlux SDR Video to HDR Ψ**. The node creates a lazy `HDR_VIDEO_STREAM`;
it does not materialize an `IMAGE` batch. **Save HDR10 Video Ψ** consumes that
stream frame by frame, reusing the first bridge-noise tensor and carrying the
previous applied RQS spline parameters forward with the upstream default 0.8
EMA. The single `shared_noise` switch enables or disables that whole temporal
bundle in both solver nodes. The saver writes HEVC Main10 `yuv420p10le` with
PQ, BT.2020, 1000-nit mastering display, MaxCLL/MaxFALL metadata, and optionally
preserved source audio; it can instead stream to float32 EXR or 16-bit PQ PNG
sequences.

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
