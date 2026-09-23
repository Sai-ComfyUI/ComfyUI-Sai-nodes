"""Native FLUX adapter runtime derived from LumaFlux (Apache-2.0).

This port keeps ComfyUI's FLUX modules—including quantized linear layers—and
replaces only block execution so the released PGA/PCM/coupler weights can run
without converting the backbone to Diffusers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import (
    luma_2020,
    rgb_to_ycbcr2020,
    sdr_to_linear2020,
    ycbcr2020_to_rgb,
)


def _physical_maps(x: torch.Tensor) -> torch.Tensor:
    y = luma_2020(x)
    kernel = y.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 4
    padded = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(padded, kernel), F.conv2d(padded, kernel.transpose(-1, -2))
    gradient = torch.log1p((gx.square() + gy.square() + 1e-12).sqrt())
    maximum, minimum = x.amax(dim=1, keepdim=True), x.amin(dim=1, keepdim=True)
    saturation = (maximum - minimum) / (maximum + 1e-6)
    return torch.cat([y, gradient, saturation], dim=1)


def _global_stats(x: torch.Tensor) -> torch.Tensor:
    flat = luma_2020(x).flatten(1)
    quantiles = torch.quantile(
        flat.float(),
        torch.tensor([0.95, 0.99], device=flat.device, dtype=torch.float32),
        dim=1,
    ).to(flat.dtype)
    return torch.stack([flat.mean(1), flat.std(1), quantiles[0], quantiles[1]], dim=1)


def _spectral_bands(y: torch.Tensor, count: int) -> torch.Tensor:
    batch, _, height, width = y.shape
    spectrum = torch.fft.rfft2(y.float(), norm="ortho").abs().square()
    fy = torch.fft.fftfreq(height, device=y.device).abs()
    fx = torch.fft.rfftfreq(width, device=y.device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    indices = ((radius / radius.max().clamp(min=1e-8)).clamp(max=1 - 1e-6) * count).long()
    flat_indices = indices.flatten().expand(batch, -1)
    result = torch.zeros(batch, count, device=y.device, dtype=spectrum.dtype)
    result.scatter_add_(1, flat_indices, spectrum.reshape(batch, -1))
    sizes = torch.zeros(count, device=y.device, dtype=spectrum.dtype)
    sizes.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten(), dtype=spectrum.dtype))
    return torch.log1p(result / sizes.clamp(min=1)).to(y.dtype)


class PhysicalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.mlp_g = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 16))

    def forward(self, sdr: torch.Tensor) -> dict[str, torch.Tensor]:
        linear = sdr_to_linear2020(sdr)
        return {
            "map": self.conv(_physical_maps(linear)),
            "g": self.mlp_g(_global_stats(linear)),
            "r": _spectral_bands(luma_2020(linear), 8),
        }


class Connector(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1152, 64), nn.SiLU(), nn.Linear(64, output_dim))


def _sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    frequencies = torch.exp(-math.log(10_000.0) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    values = t.float()[:, None] * frequencies[None] * 1000.0
    return torch.cat([values.sin(), values.cos()], dim=-1).to(t.dtype)


class TimestepLayerModulation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t_proj = nn.Linear(128, 128)
        self.layer_embed = nn.Embedding(57, 128)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.head = nn.Linear(128, 6)

    def forward(self, t: torch.Tensor, index: int) -> dict[str, torch.Tensor]:
        layer = torch.full_like(t, index, dtype=torch.long)
        values = self.head(self.mlp(self.t_proj(_sinusoidal_embedding(t)) + self.layer_embed(layer)))
        pga, pga_bias, pcm, pcm_bias, spectral, coupler = values.unbind(-1)
        return {
            "alpha_pga": pga.sigmoid(), "beta_pga": pga_bias,
            "alpha_pcm": pcm.sigmoid(), "beta_pcm": pcm_bias,
            "n_spec": F.softplus(spectral), "lam": coupler.sigmoid(),
        }


class PGA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Linear(3072, 8, bias=False)
        self.up = nn.Linear(8, 3072, bias=False)
        self.p_v = nn.Linear(48, 24)
        self.w_r = nn.Linear(8, 24)

    def forward(self, x, value, physical, stats, bands, schedule, prefix: int):
        batch, tokens, _ = x.shape
        residual = schedule["alpha_pga"].view(batch, 1, 1) * self.up(self.down(x))
        residual = residual + schedule["beta_pga"].view(batch, 1, 1) * x
        condition = torch.cat([physical, stats[:, None].expand(-1, physical.shape[1], -1)], -1)
        gate = torch.sigmoid(self.p_v(condition))
        if prefix:
            gate = torch.cat([gate.new_ones(batch, prefix, 24), gate], dim=1)
        spectral = F.softplus(self.w_r(bands))
        gate = gate * (1 + schedule["n_spec"].view(batch, 1, 1) * spectral[:, None])
        residual = residual.view(batch, tokens, 24, 128) * gate[:, :tokens, :, None]
        return value + residual.reshape(batch, tokens, 3072)


class PCM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(3072, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(3072, 64), nn.SiLU(), nn.Linear(64, 6144))

    def forward(self, hidden, perceptual, schedule):
        batch = hidden.shape[0]
        coeff = schedule["alpha_pcm"].view(batch, 1) * self.mlp(perceptual.mean(1))
        coeff = coeff + schedule["beta_pcm"].view(batch, 1)
        gamma, offset = coeff.chunk(2, -1)
        return hidden + gamma[:, None] * self.norm(hidden) + offset[:, None]


def _resample(tokens: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, count, dim = tokens.shape
    side = int(math.sqrt(count))
    if side * side != count:
        return tokens.mean(1, keepdim=True).expand(batch, size[0] * size[1], dim)
    grid = tokens.transpose(1, 2).reshape(batch, dim, side, side)
    return F.interpolate(grid, size=size, mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)


class Coupler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_p = nn.Sequential(nn.Linear(32, 64, bias=False), nn.SiLU(), nn.Linear(64, 3072, bias=False))
        self.w_c = nn.Sequential(nn.Linear(3072, 64, bias=False), nn.SiLU(), nn.Linear(64, 3072, bias=False))

    def forward(self, hidden, physical, perceptual, schedule, grid):
        fused = self.w_p(physical) + self.w_c(_resample(perceptual, grid))
        return hidden + schedule["lam"].view(-1, 1, 1) * fused


class BlockAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pga, self.pcm, self.coupler = PGA(), PCM(), Coupler()


class RQSToneDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(16, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 25))
        self.chroma = nn.Conv2d(2, 2, 1)
        self.register_buffer("peak_pq", torch.tensor(0.7518270962))

    @staticmethod
    def _spline(y, widths, heights, derivatives):
        batch, bins = widths.shape
        widths = 1e-3 + (1 - 1e-3 * bins) * F.softmax(widths, -1)
        heights = 1e-3 + (1 - 1e-3 * bins) * F.softmax(heights, -1)
        derivatives = 1e-3 + F.softplus(derivatives)
        cum_w, cum_h = F.pad(widths.cumsum(-1), (1, 0)), F.pad(heights.cumsum(-1), (1, 0))
        cum_w[..., -1] = cum_h[..., -1] = 1.0
        flat = y.clamp(0, 1).reshape(batch, -1)
        index = torch.searchsorted(cum_w[..., 1:-1].contiguous(), flat.contiguous()).clamp(max=bins - 1)
        in_w, left_w = widths.gather(-1, index), cum_w.gather(-1, index)
        in_h, left_h = heights.gather(-1, index), cum_h.gather(-1, index)
        d0, d1 = derivatives.gather(-1, index), derivatives.gather(-1, index + 1)
        slope = in_h / in_w
        theta = ((flat - left_w) / in_w).clamp(0, 1)
        mix = theta * (1 - theta)
        numerator = in_h * (slope * theta.square() + d0 * mix)
        denominator = slope + (d0 + d1 - 2 * slope) * mix
        return (left_h + numerator / denominator.clamp(min=1e-8)).reshape_as(y).clamp(0, 1)

    def spline_params(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.head(latent.float().mean((-2, -1)).to(self.head[0].weight.dtype))
        return params[:, :8], params[:, 8:16], params[:, 16:]

    def forward(
        self,
        decoded: torch.Tensor,
        latent: torch.Tensor,
        strength: float,
        spline_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        widths, heights, derivatives = spline_params or self.spline_params(latent)
        peak = self.peak_pq.to(decoded)
        y, cb, cr = rgb_to_ycbcr2020(decoded.clamp(0, peak)).split(1, 1)
        adjusted = self._spline(y / peak, widths, heights, derivatives) * peak
        adjusted = y + strength * (adjusted - y)
        chroma = self.chroma(torch.cat([cb, cr], 1))
        return ycbcr2020_to_rgb(torch.cat([adjusted, chroma], 1)).clamp(0, peak)


@dataclass
class RuntimeCondition:
    physical: torch.Tensor
    stats: torch.Tensor
    bands: torch.Tensor
    perceptual: torch.Tensor
    context: torch.Tensor
    pooled: torch.Tensor
    time: torch.Tensor | None = None
    grid: tuple[int, int] | None = None


class LumaFluxAdapter(nn.Module):
    """Released LumaFlux trainable modules plus native Comfy FLUX callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.physical = PhysicalEncoder()
        self.perc_connector = Connector(3072)
        self.ctx_connector = Connector(4096)
        self.null_context = nn.Parameter(torch.empty(8, 4096))
        self.null_pooled = nn.Parameter(torch.empty(768))
        self.pooled_from_stats = nn.Linear(16, 768)
        self.modulation = TimestepLayerModulation()
        self.double_blocks = nn.ModuleList(BlockAdapter() for _ in range(19))
        self.single_blocks = nn.ModuleList(BlockAdapter() for _ in range(38))
        self.rqs = RQSToneDecoder()
        self.condition: RuntimeCondition | None = None

    def prepare_condition(
        self,
        sdr: torch.Tensor,
        siglip_tokens: torch.Tensor,
        grid: tuple[int, int],
    ) -> RuntimeCondition:
        features = self.physical(sdr)
        batch = sdr.shape[0]
        physical = F.adaptive_avg_pool2d(features["map"], grid).flatten(2).transpose(1, 2)
        perceptual = self.perc_connector.proj(siglip_tokens)
        context = torch.cat([self.null_context[None].expand(batch, -1, -1), self.ctx_connector.proj(siglip_tokens)], 1)
        pooled = self.null_pooled[None].expand(batch, -1) + self.pooled_from_stats(features["g"])
        return RuntimeCondition(
            physical,
            features["g"],
            features["r"],
            perceptual,
            context,
            pooled,
            grid=grid,
        )

    def _schedule(self, index: int):
        if self.condition is None or self.condition.time is None:
            raise RuntimeError("LumaFlux conditioning is not active")
        return self.modulation(self.condition.time, index)

    def _run_attention_patches(self, q, k, v, pe, mask, options):
        for patch in options.get("patches", {}).get("attn1_patch", []):
            result = patch(q, k, v, pe=pe, attn_mask=mask, extra_options=options.copy())
            q, k, v = result.get("q", q), result.get("k", k), result.get("v", v)
            pe, mask = result.get("pe", pe), result.get("attn_mask", mask)
        return q, k, v, pe, mask

    def double_block(self, base, index: int, args: dict) -> dict:
        from comfy.ldm.flux.layers import apply_mod
        from comfy.ldm.flux.math import attention

        condition, adapter = self.condition, self.double_blocks[index]
        assert condition is not None and condition.grid is not None
        schedule = self._schedule(index)
        img, txt, vec, pe = args["img"], args["txt"], args["vec"], args["pe"]
        img = adapter.pcm(img, condition.perceptual, schedule)
        img_mod1, img_mod2 = base.img_mod(vec)
        txt_mod1, txt_mod2 = base.txt_mod(vec)
        img_normal = apply_mod(base.img_norm1(img), 1 + img_mod1.scale, img_mod1.shift)
        txt_normal = apply_mod(base.txt_norm1(txt), 1 + txt_mod1.scale, txt_mod1.shift)
        img_qkv = base.img_attn.qkv(img_normal).view(img.shape[0], img.shape[1], 3, base.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_qkv = base.txt_attn.qkv(txt_normal).view(txt.shape[0], txt.shape[1], 3, base.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = base.img_attn.norm(img_qkv[0], img_qkv[1], img_qkv[2])
        txt_q, txt_k = base.txt_attn.norm(txt_qkv[0], txt_qkv[1], txt_qkv[2])
        q, k = torch.cat([txt_q, img_q], 2), torch.cat([txt_k, img_k], 2)
        value = torch.cat([txt_qkv[2], img_qkv[2]], 2).permute(0, 2, 1, 3).reshape(img.shape[0], -1, 3072)
        source = torch.cat([txt_normal, img_normal], 1)
        value = adapter.pga(source, value, condition.physical, condition.stats, condition.bands, schedule, txt.shape[1])
        v = value.view(img.shape[0], -1, 24, 128).permute(0, 2, 1, 3)
        options = args["transformer_options"]
        options = {**options, "img_slice": [txt.shape[1], q.shape[2]]}
        q, k, v, pe, mask = self._run_attention_patches(q, k, v, pe, args.get("attn_mask"), options)
        attended = attention(q, k, v, pe=pe, mask=mask, transformer_options=options)
        for patch in options.get("patches", {}).get("attn1_output_patch", []):
            attended = patch(attended, options.copy())
        txt_attn, img_attn = attended[:, :txt.shape[1]], attended[:, txt.shape[1]:]
        img = img + apply_mod(base.img_attn.proj(img_attn), img_mod1.gate)
        img = img + apply_mod(base.img_mlp(apply_mod(base.img_norm2(img), 1 + img_mod2.scale, img_mod2.shift)), img_mod2.gate)
        txt = txt + apply_mod(base.txt_attn.proj(txt_attn), txt_mod1.gate)
        txt = txt + apply_mod(base.txt_mlp(apply_mod(base.txt_norm2(txt), 1 + txt_mod2.scale, txt_mod2.shift)), txt_mod2.gate)
        img = adapter.coupler(img, condition.physical, condition.perceptual, schedule, condition.grid)
        return {"img": img, "txt": txt}

    def single_block(self, base, index: int, args: dict) -> dict:
        from comfy.ldm.flux.layers import apply_mod
        from comfy.ldm.flux.math import attention

        condition, adapter = self.condition, self.single_blocks[index]
        assert condition is not None and condition.grid is not None
        schedule = self._schedule(19 + index)
        hidden, vec, pe = args["img"], args["vec"], args["pe"]
        options = args["transformer_options"]
        start, end = options["img_slice"]
        image = adapter.pcm(hidden[:, start:end], condition.perceptual, schedule)
        hidden = torch.cat([hidden[:, :start], image, hidden[:, end:]], 1)
        modulation, _ = base.modulation(vec)
        normal = apply_mod(base.pre_norm(hidden), 1 + modulation.scale, modulation.shift)
        qkv, mlp = torch.split(base.linear1(normal), [3 * base.hidden_size, base.mlp_hidden_dim_first], -1)
        q, k, v = qkv.view(hidden.shape[0], hidden.shape[1], 3, base.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = base.norm(q, k, v)
        value = v.permute(0, 2, 1, 3).reshape(hidden.shape[0], hidden.shape[1], 3072)
        value = adapter.pga(normal, value, condition.physical, condition.stats, condition.bands, schedule, start)
        v = value.view(hidden.shape[0], hidden.shape[1], 24, 128).permute(0, 2, 1, 3)
        q, k, v, pe, mask = self._run_attention_patches(q, k, v, pe, args.get("attn_mask"), options)
        attended = attention(q, k, v, pe=pe, mask=mask, transformer_options=options)
        for patch in options.get("patches", {}).get("attn1_output_patch", []):
            attended = patch(attended, options.copy())
        mlp = base.mlp_act(mlp)
        hidden = hidden + apply_mod(base.linear2(torch.cat([attended, mlp], 2)), modulation.gate)
        image = adapter.coupler(hidden[:, start:end], condition.physical, condition.perceptual, schedule, condition.grid)
        return {"img": torch.cat([hidden[:, :start], image, hidden[:, end:]], 1)}


_DOUBLE = re.compile(r"^transformer\.transformer_blocks\.(\d+)\.(.+)$")
_SINGLE = re.compile(r"^transformer\.single_transformer_blocks\.(\d+)\.(.+)$")


def remap_checkpoint_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map released Diffusers wrapper keys to the native adapter module."""
    mapped = {}
    for key, value in state.items():
        double = _DOUBLE.match(key)
        single = _SINGLE.match(key)
        if double:
            index, suffix = double.groups()
            if suffix.startswith("modulation."):
                mapped[suffix] = value
            else:
                mapped[f"double_blocks.{index}.{suffix}"] = value
        elif single:
            index, suffix = single.groups()
            if not suffix.startswith("modulation."):
                mapped[f"single_blocks.{index}.{suffix}"] = value
        else:
            mapped[key] = value
    return mapped
