# Third-party notices

This repository contains integrations with and modified source derived from
third-party projects. Those components retain their original copyrights and
licenses.

## MyTimeMachine

`vendor/mytimemachine/` is derived from
[`luchaoqi/mytimemachine`](https://github.com/luchaoqi/mytimemachine), source
revision `ed6111fe7d5d1424215d6cad77b30c898b9b9f86`, and includes modified
inference and training code. It is distributed under the MIT license in
[`vendor/mytimemachine/LICENSE`](vendor/mytimemachine/LICENSE). Additional
upstream notices for embedded and referenced implementations are retained under
[`vendor/mytimemachine/licenses/`](vendor/mytimemachine/licenses/).

MyTimeMachine model weights are not included. Their original download sources
and purposes are documented in [`docs/MODELS.md`](docs/MODELS.md).

## LUA-FLUX

The LUA-FLUX loader and latent-upscaler interoperate with the model from
[`vaskers5/LUA-FLUX`](https://huggingface.co/vaskers5/LUA-FLUX). Related
architecture license text is retained in
[`nodes/latent/LUA_LICENSE`](nodes/latent/LUA_LICENSE). The model checkpoint is
downloaded separately and is not included in this repository.

## OpenCV Zoo YuNet

The YuNet detector model is downloaded separately from
[`opencv/opencv_zoo`](https://github.com/opencv/opencv_zoo). Its license text is
retained in [`models/mytimemachine/YUNET_LICENSE`](models/mytimemachine/YUNET_LICENSE).

## Fonts

No font binaries are bundled. The collage node discovers fonts installed on the
host system; see [`assets/fonts/README.md`](assets/fonts/README.md).
