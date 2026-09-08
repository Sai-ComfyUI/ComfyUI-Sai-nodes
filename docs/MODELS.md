# Model downloads

Model weights are not distributed in the Git repository. Download only from
sources you trust: PyTorch `.pt` and `.pth` checkpoints can contain executable
pickle data.

## MyTimeMachine directory

Place MyTimeMachine-related files here:

```text
ComfyUI/custom_nodes/ComfyUI-Sai-nodes/models/mytimemachine/
├── sam_ffhq_aging.pt
├── face_detection_yunet_2026may.onnx
├── psp_ffhq_encode.pt
├── stylegan2-ffhq-config-f.pt
├── model_ir_se50.pth
├── dex_age_classifier.pth
├── lpips_alex_v0.1.pth
└── personalized/
    └── your_person_checkpoint.pt
```

The loader scans subdirectories. The repository ignores checkpoint and ONNX
files, so they remain local.

## Files required by feature

| Feature | Required files |
| --- | --- |
| General re-ageing | `sam_ffhq_aging.pt` |
| Personalized re-ageing | `sam_ffhq_aging.pt` and a personalized checkpoint |
| YuNet face align/restore | `face_detection_yunet_2026may.onnx` |
| Personalized training | `sam_ffhq_aging.pt`, `psp_ffhq_encode.pt`, `stylegan2-ffhq-config-f.pt`, `model_ir_se50.pth`, `dex_age_classifier.pth`, `lpips_alex_v0.1.pth` |

## Primary sources

The upstream [MyTimeMachine repository](https://github.com/luchaoqi/mytimemachine)
publishes the first five model sources in its Model Zoo:

| Local filename | Original source | Purpose |
| --- | --- | --- |
| `sam_ffhq_aging.pt` | [MyTimeMachine/SAM checkpoint](https://drive.google.com/file/d/1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC/view?usp=sharing) | General aging model and global aging prior |
| `psp_ffhq_encode.pt` | [pSp Encoder](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing) | StyleGAN inversion initialization for training |
| `stylegan2-ffhq-config-f.pt` | [FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | FFHQ generator initialization for training |
| `model_ir_se50.pth` | [IR-SE50 model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Identity loss during training |
| `dex_age_classifier.pth` | [VGG age classifier](https://drive.google.com/file/d/1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh/view?usp=sharing) | Aging loss during training |
| `lpips_alex_v0.1.pth` | [LPIPS AlexNet weights](https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth) | Perceptual loss during training; rename `alex.pth` after downloading |
| `face_detection_yunet_2026may.onnx` | [OpenCV Zoo YuNet](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx) | Face detection and five-point alignment |

## Verified mirror

As a secondary source, all seven files are available from
[`sailing/ComfyUI-Sai-nodes_Models`](https://huggingface.co/sailing/ComfyUI-Sai-nodes_Models/tree/main/mytimemachine).
The repository listing and direct `resolve` endpoints were verified on
2026-09-08. The set is approximately 5.65 GB and matches the filenames expected
by the nodes.

## LUA-FLUX

**Load LUA FLUX Model Ψ** downloads `lua_flux.pth` from the original
[`vaskers5/LUA-FLUX`](https://huggingface.co/vaskers5/LUA-FLUX) repository into
ComfyUI's standard `models/latent_upscale_models` directory. It can also load an
existing file from that directory.
