# LumaFlux native ComfyUI port

This directory contains a modified, ComfyUI-native inference implementation
derived from [shreshthsaini/LumaFlux](https://github.com/shreshthsaini/LumaFlux),
source revision `bfd2f0eb4c53104f5c61ad85f728c6e0e2bd678b`.

The original work and these derived components are used under the Apache
License 2.0. A complete copy of that license is retained in
`nodes/latent/LUA_LICENSE` at the repository root. The native port changes the
Diffusers FLUX wrappers into ComfyUI block replacements, retains ComfyUI model
and quantization management, and adds ComfyUI data-contract integration.

LumaFlux adapter weights and FLUX.1-dev weights are not part of this source
distribution. They retain their respective upstream terms. In particular,
FLUX.1-dev is subject to its own non-commercial model license.
