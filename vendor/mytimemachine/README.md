# Vendored MyTimeMachine inference and training architecture

These files are derived from <https://github.com/luchaoqi/mytimemachine> and its
embedded rosinality StyleGAN2 implementation. See `LICENSE` and the upstream
license notices.

Local changes:

- package-relative imports;
- a decoder-free encoding method for the global prior;
- registered latent-average buffer for ComfyUI device movement;
- native PyTorch inference fallbacks instead of import-time legacy CUDA builds.
- personalized-training runner with configurable local paths and exact final
  checkpoint destination;
- background job state reporting, managed LPIPS weights, Windows-safe debug
  paths, reduced unused gradients, and a corrected maximum-step exit condition.

Source revision: `ed6111fe7d5d1424215d6cad77b30c898b9b9f86`.
