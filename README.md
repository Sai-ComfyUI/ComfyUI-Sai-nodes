# ComfyUI-Sai-nodes

A ComfyUI V3 custom-node collection for conditioning, latent upscaling, face
alignment, personalized re-ageing, and labeled image comparison.

[繁體中文說明](README.zh-TW.md)

## Included features

- **Multi Reference Latent Ψ** prepares multiple image references for edit
  models and passes through the VAE.
- **LUA FLUX** loads the official LUA-FLUX checkpoint and performs x2 or x4
  latent upscaling for FLUX.1.
- **MyTimeMachine** provides model loading, FFHQ face alignment/restoration,
  general or personalized facial re-ageing, and personalized training.
- **Labeled Image Collage Ψ** builds labeled comparison grids with wrapping,
  typography, transparency, and browser-local named presets.
- A frontend restart action is available for supervisors that restart ComfyUI
  after exit code `75`.

See [Usage](docs/USAGE.md) for the complete node list and workflow guidance.

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Sai-ComfyUI/ComfyUI-Sai-nodes.git
```

Install dependencies with the same Python interpreter that runs ComfyUI:

```bash
python -m pip install -r ComfyUI-Sai-nodes/requirements.txt
```

For the Windows portable build, run this from the portable installation root:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Sai-nodes\requirements.txt
```

Restart ComfyUI after installation. A recent ComfyUI version with the V3
`comfy_api.latest` API is required. See the detailed
[installation and update guide](docs/INSTALLATION.md) for alternate layouts and
troubleshooting.

## Models

Model weights are intentionally not stored in this Git repository.

- LUA-FLUX can be downloaded by its loader node from the official
  [`vaskers5/LUA-FLUX`](https://huggingface.co/vaskers5/LUA-FLUX) repository.
- MyTimeMachine weights belong under
  `ComfyUI-Sai-nodes/models/mytimemachine/`. Use the original projects as the
  primary sources. A verified mirror is available at
  [`sailing/ComfyUI-Sai-nodes_Models`](https://huggingface.co/sailing/ComfyUI-Sai-nodes_Models/tree/main/mytimemachine).

The exact files required for inference, face alignment, and training are listed
in [Model downloads](docs/MODELS.md).

## Documentation

- [Installation and updates](docs/INSTALLATION.md)
- [Model downloads and directory layout](docs/MODELS.md)
- [Node usage](docs/USAGE.md)
- Individual node help pages are in [`docs/`](docs/).
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

Project code is released under the [MIT License](LICENSE). Vendored code and
external model files retain their original licenses; see
[Third-party notices](THIRD_PARTY_NOTICES.md).
