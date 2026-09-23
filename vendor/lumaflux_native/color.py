"""LumaFlux HDR color helpers (Apache-2.0 upstream, revision bfd2f0e)."""

from __future__ import annotations

import torch

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0
PQ_PEAK_NITS = 10_000.0

M_2020_TO_709 = torch.tensor(
    [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]]
)
M_2020_TO_709 = M_2020_TO_709 / M_2020_TO_709.sum(dim=1, keepdim=True)
M_709_TO_2020 = torch.linalg.inv(M_2020_TO_709.to(torch.float64)).float()
M_2020_TO_XYZ = torch.tensor(
    [[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0, 0.0281, 1.0610]]
)
M_XYZ_TO_2020 = torch.linalg.inv(M_2020_TO_XYZ.to(torch.float64)).float()


def apply_matrix(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(dtype=x.dtype, device=x.device)
    return (x.movedim(-3, -1) @ matrix.T).movedim(-1, -3)


def pq_eotf_nits(signal: torch.Tensor) -> torch.Tensor:
    encoded = signal.clamp(1e-8, 1.0).pow(1.0 / PQ_M2)
    linear = ((encoded - PQ_C1).clamp(min=0.0) / (PQ_C2 - PQ_C3 * encoded)).pow(1.0 / PQ_M1)
    return linear * PQ_PEAK_NITS


def pq_oetf_nits(nits: torch.Tensor) -> torch.Tensor:
    linear = (nits / PQ_PEAK_NITS).clamp(1e-10, 1.0)
    powered = linear.pow(PQ_M1)
    return ((PQ_C1 + PQ_C2 * powered) / (1.0 + PQ_C3 * powered)).pow(PQ_M2)


def bt1886_eotf(signal: torch.Tensor) -> torch.Tensor:
    return signal.clamp(0.0, 1.0).pow(2.4)


def bt1886_inverse_eotf(linear: torch.Tensor) -> torch.Tensor:
    return linear.clamp(0.0, 1.0).pow(1.0 / 2.4)


def sdr_to_linear2020(x_sdr: torch.Tensor) -> torch.Tensor:
    return apply_matrix(bt1886_eotf(x_sdr), M_709_TO_2020).clamp(0.0, 1.0)


def luma_2020(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.2627, 0.6780, 0.0593]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=-3, keepdim=True)


def rgb_to_ycbcr2020(rgb: torch.Tensor) -> torch.Tensor:
    red, green, blue = rgb.unbind(dim=-3)
    y = 0.2627 * red + 0.6780 * green + 0.0593 * blue
    return torch.stack([y, (blue - y) / 1.8814, (red - y) / 1.4746], dim=-3)


def ycbcr2020_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr.unbind(dim=-3)
    red = y + 1.4746 * cr
    blue = y + 1.8814 * cb
    green = (y - 0.2627 * red - 0.0593 * blue) / 0.6780
    return torch.stack([red, green, blue], dim=-3)


def _scale_by_luma(x: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scaled = x * (target / source.clamp(min=1e-8))
    return torch.where(source > 0.0, scaled, torch.zeros_like(scaled))


def hdr_to_sdr_bt2446c(x_pq: torch.Tensor, peak_nits: float = 1000.0, alpha: float = 0.10) -> torch.Tensor:
    """Display-referred PQ/BT.2020 to BT.709 preview using BT.2446 Method C."""
    if not 0.0 < peak_nits <= 10_000.0:
        raise ValueError("peak_nits must be in (0, 10000]")
    if not 0.0 <= alpha < 1.0 / 3.0:
        raise ValueError("alpha must be in [0, 1/3)")
    nits = pq_eotf_nits(x_pq).clamp(0.0, peak_nits)
    cross = nits.new_tensor(
        [[1 - 2 * alpha, alpha, alpha], [alpha, 1 - 2 * alpha, alpha], [alpha, alpha, 1 - 2 * alpha]]
    )
    rgb_x = apply_matrix(nits, cross)
    xyz = apply_matrix(rgb_x, M_2020_TO_XYZ)
    source_y = xyz[:, 1:2]
    inflection = 58.5 / 0.83802

    def curve(value: torch.Tensor) -> torch.Tensor:
        low = 0.83802 * value
        high = 15.09968 * torch.log((value / inflection - 0.74204).clamp(min=1e-12)) + 78.99439
        return torch.where(value < inflection, low, high)

    scale = 100.0 / curve(nits.new_tensor(peak_nits))
    xyz_sdr = _scale_by_luma(xyz, source_y, curve(source_y) * scale)
    rgb_x_sdr = apply_matrix(xyz_sdr, M_XYZ_TO_2020)
    rgb_2020 = apply_matrix(rgb_x_sdr, torch.linalg.inv(cross)) / 100.0
    rgb_709 = apply_matrix(rgb_2020, M_2020_TO_709).clamp(0.0, 1.0)
    return bt1886_inverse_eotf(rgb_709)


def hdr_to_sdr_whitepoint(
    x_pq: torch.Tensor,
    white_nits: float = 203.0,
    peak_nits: float = 1000.0,
    method: str = "reinhard",
) -> torch.Tensor:
    """PQ/BT.2020 to a BT.709/BT.1886 preview with an adjustable SDR white."""
    if not 0.0 < white_nits <= peak_nits <= 10_000.0:
        raise ValueError("white_nits and peak_nits must satisfy 0 < white_nits <= peak_nits <= 10000")
    linear = apply_matrix(pq_eotf_nits(x_pq), M_2020_TO_709).clamp(min=0.0) / white_nits
    if method == "reinhard":
        white_level = max(peak_nits / white_nits, 1.0)
        y = 0.2126 * linear[:, 0:1] + 0.7152 * linear[:, 1:2] + 0.0722 * linear[:, 2:3]
        mapped_y = y * (1.0 + y / (white_level * white_level)) / (1.0 + y)
        linear = linear * (mapped_y / y.clamp(min=1e-6))
    elif method != "clip":
        raise ValueError(f"Unsupported SDR preview method: {method}")
    return bt1886_inverse_eotf(linear.clamp(0.0, 1.0))
