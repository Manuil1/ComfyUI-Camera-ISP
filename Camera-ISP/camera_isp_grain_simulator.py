"""
Camera ISP grain simulator

ComfyUI custom node for camera-style post processing:
- Bayer CFA simulation on a single raw channel.
- Normalized convolution demosaicing.
- Deterministic PRNU, shot noise, read noise, vignette and luma-aware grain.
- Optional mask and diagnostic delta output.

This node is intended for transparent photographic realism, forensic-style
experimentation, and graphics/post-production work. It is not a detector bypass
or impersonation tool.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


_EPS = 1.0e-6
_SEED_MOD = (2**63) - 1


_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "neutral": {
        "shot": 1.0,
        "read": 1.0,
        "prnu": 1.0,
        "vignette": 1.0,
        "grain": 1.0,
        "chroma": 1.0,
    },
    "iphone_12": {
        "shot": 1.15,
        "read": 1.10,
        "prnu": 1.05,
        "vignette": 1.10,
        "grain": 0.95,
        "chroma": 1.12,
    },
    "iphone_15": {
        "shot": 0.92,
        "read": 0.88,
        "prnu": 0.85,
        "vignette": 0.88,
        "grain": 0.85,
        "chroma": 0.80,
    },
    "full_frame": {
        "shot": 0.58,
        "read": 0.62,
        "prnu": 0.65,
        "vignette": 0.70,
        "grain": 0.70,
        "chroma": 0.55,
    },
}


def _safe_seed(seed: int, offset: int = 0) -> int:
    return (int(seed) + int(offset)) % _SEED_MOD


def _randn(
    shape: Tuple[int, ...],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    seed = _safe_seed(seed)
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    except Exception:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=dtype
        )


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("Expected ComfyUI IMAGE tensor with shape [B,H,W,C].")
    if image.shape[-1] < 3:
        image = image.repeat(1, 1, 1, 3)
    return image[..., :3].permute(0, 3, 1, 2).contiguous()


def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 2, 3, 1).contiguous()


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    low = x / 12.92
    high = torch.pow(((x + 0.055) / 1.055).clamp_min(0.0), 2.4)
    return torch.where(x <= 0.04045, low, high)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    low = x * 12.92
    high = 1.055 * torch.pow(x.clamp_min(0.0), 1.0 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, low, high).clamp(0.0, 1.0)


def _gaussian_kernel1d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = max(float(sigma), 0.01)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(_EPS)


def _separable_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x

    b, c, h, w = x.shape
    kernel = _gaussian_kernel1d(sigma, x.device, x.dtype)
    radius = kernel.numel() // 2

    if w > 1:
        mode = "reflect" if w > radius else "replicate"
        weight_x = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        x = F.pad(x, (radius, radius, 0, 0), mode=mode)
        x = F.conv2d(x, weight_x, groups=c)

    if h > 1:
        mode = "reflect" if h > radius else "replicate"
        weight_y = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        x = F.pad(x, (0, 0, radius, radius), mode=mode)
        x = F.conv2d(x, weight_y, groups=c)

    return x


def _development_power(
    luma: torch.Tensor,
    shadow_level: float = 0.15,
    high_level: float = 0.25,
) -> torch.Tensor:
    x = (luma * 255.0).clamp(0.0, 255.0)
    power = torch.full_like(x, 0.5)

    shadow = x < 160.0
    highlight = x >= 200.0
    shadow_power = 0.5 - (160.0 - x) * (0.5 - shadow_level) / 160.0
    high_power = 0.5 - (x - 200.0) * (0.5 - high_level) / 55.0

    power = torch.where(shadow, shadow_power, power)
    power = torch.where(highlight, high_power, power)
    return power.clamp(0.05, 1.0)


def _profile_from_inputs(profile: str, profile_path: Optional[str]) -> Dict[str, float]:
    search = f"{profile or ''} {profile_path or ''}".lower()
    for key in ("iphone_15", "iphone_12", "full_frame", "neutral"):
        if key in search:
            return _PROFILE_PRESETS[key]
    return _PROFILE_PRESETS.get(profile, _PROFILE_PRESETS["neutral"])


def _bayer_masks(
    h: int,
    w: int,
    pattern: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pattern = pattern.upper()
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        pattern = "RGGB"

    channel_for = {"R": 0, "G": 1, "B": 2}
    positions = (
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    )

    masks = torch.zeros((1, 3, h, w), device=device, dtype=dtype)
    for row, col, color in positions:
        masks[:, channel_for[color], row::2, col::2] = 1.0
    return masks


def _rgb_to_bayer_raw(x: torch.Tensor, pattern: str) -> torch.Tensor:
    b, _c, h, w = x.shape
    pattern = pattern.upper()
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        pattern = "RGGB"

    channel_for = {"R": 0, "G": 1, "B": 2}
    positions = (
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    )

    raw = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
    for row, col, color in positions:
        raw[:, 0, row::2, col::2] = x[:, channel_for[color], row::2, col::2]
    return raw


def _demosaic_bilinear(raw: torch.Tensor, pattern: str) -> torch.Tensor:
    _b, _c, h, w = raw.shape
    masks = _bayer_masks(h, w, pattern, raw.device, raw.dtype)
    samples = raw.repeat(1, 3, 1, 1) * masks

    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=raw.device,
        dtype=raw.dtype,
    ).view(1, 1, 3, 3)
    weight = kernel.repeat(3, 1, 1, 1)

    numer = F.conv2d(samples, weight, padding=1, groups=3)
    denom = F.conv2d(masks, weight, padding=1, groups=3)
    return numer / denom.clamp_min(_EPS)


def _prepare_mask(
    mask: Optional[torch.Tensor],
    shape: Tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[-1] <= 4:
        mask = mask[..., :1].permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError("Unsupported MASK shape.")

    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[-2:] != shape[-2:]:
        mask = F.interpolate(mask, size=shape[-2:], mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _prepare_development_curve(
    curve: Optional[torch.Tensor],
    shape: Tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if curve is None:
        return None

    curve_bchw = _to_bchw(curve).to(device=device, dtype=dtype)
    if curve_bchw.shape[1] == 3:
        curve_bchw = (
            0.299 * curve_bchw[:, 0:1]
            + 0.587 * curve_bchw[:, 1:2]
            + 0.114 * curve_bchw[:, 2:3]
        )

    if curve_bchw.shape[-2:] != shape[-2:]:
        curve_bchw = F.interpolate(
            curve_bchw, size=shape[-2:], mode="bilinear", align_corners=False
        )
    return curve_bchw.clamp(0.0, 2.0)


def _multiscale_noise(
    b: int,
    c: int,
    h: int,
    w: int,
    seed: int,
    grain_size: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    grain_size = max(float(grain_size), 0.5)
    if grain_size > 1.05:
        small_h = max(1, int(math.ceil(h / grain_size)))
        small_w = max(1, int(math.ceil(w / grain_size)))
        noise = _randn((b, c, small_h, small_w), seed, device, dtype)
        noise = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
    else:
        noise = _randn((b, c, h, w), seed, device, dtype)

    noise = noise - noise.mean(dim=(-2, -1), keepdim=True)
    std = noise.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-4)
    return noise / std


def _chromatic_aberration(x: torch.Tensor, amount_px: float) -> torch.Tensor:
    amount_px = float(amount_px)
    if amount_px <= 0.0 or x.shape[1] < 3:
        return x

    b, _c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    scale = amount_px * 2.0 / max(float(max(h, w)), 1.0)
    grid_r = (base + base * scale).expand(b, -1, -1, -1)
    grid_b = (base - base * scale).expand(b, -1, -1, -1)

    red = F.grid_sample(
        x[:, 0:1],
        grid_r,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    blue = F.grid_sample(
        x[:, 2:3],
        grid_b,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return torch.cat((red, x[:, 1:2], blue), dim=1)


class CameraISPEngine:
    def __init__(self) -> None:
        self._prnu_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def _prnu_base(
        self,
        tile_size: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (str(device), int(tile_size), _safe_seed(seed, 91001))
        cached = self._prnu_cache.get(key)
        if cached is not None:
            return cached.to(dtype=dtype)

        base = _randn(
            (1, 1, int(tile_size), int(tile_size)),
            key[2],
            device,
            torch.float32,
        )
        base = _separable_blur(base, sigma=0.65)
        base = base - base.mean()
        base = base / base.std().clamp_min(1.0e-4)

        if len(self._prnu_cache) >= 4:
            oldest_key = next(iter(self._prnu_cache))
            del self._prnu_cache[oldest_key]
        self._prnu_cache[key] = base
        return base.to(dtype=dtype)

    def _prnu_map(
        self,
        h: int,
        w: int,
        tile_size: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = self._prnu_base(tile_size, seed, device, dtype)
        y = torch.arange(h, device=device) % int(tile_size)
        x = torch.arange(w, device=device) % int(tile_size)
        return base.index_select(2, y).index_select(3, x)

    def apply(
        self,
        image: torch.Tensor,
        seed: int,
        profile: str,
        profile_path: str,
        iso: int,
        bayer_pattern: str,
        prnu_strength: float,
        shot_noise: float,
        read_noise: float,
        grain_strength: float,
        grain_size: float,
        chroma_noise: float,
        vignette: float,
        chromatic_aberration: float,
        micro_contrast: float,
        prnu_tile_size: int,
        development_curve: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = image.to(dtype=torch.float32).clamp(0.0, 1.0)
        b, _c, h, w = x.shape
        device = x.device
        dtype = x.dtype
        preset = _profile_from_inputs(profile, profile_path)

        iso_scale = math.sqrt(max(float(iso), 1.0) / 100.0)
        linear = _srgb_to_linear(x)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        radius2 = xx.square() + yy.square()
        vignette_map = (1.0 - float(vignette) * preset["vignette"] * radius2).clamp(
            0.20, 1.25
        )
        vignette_map = vignette_map.unsqueeze(0).unsqueeze(0)

        bayer_enabled = bayer_pattern.lower() != "off"
        prnu = self._prnu_map(
            h,
            w,
            int(prnu_tile_size),
            seed,
            device,
            dtype,
        )

        if bayer_enabled:
            raw = _rgb_to_bayer_raw(linear, bayer_pattern)
            raw = raw * vignette_map
            raw = raw * (1.0 + prnu * float(prnu_strength) * preset["prnu"])

            shot = _randn(raw.shape, _safe_seed(seed, 17), device, dtype)
            read = _randn(raw.shape, _safe_seed(seed, 29), device, dtype)
            shot_amp = float(shot_noise) * iso_scale * preset["shot"]
            read_amp = float(read_noise) * (iso_scale**1.15) * preset["read"]
            raw = raw + shot * torch.sqrt(raw.clamp_min(0.0) + 1.0e-4) * shot_amp
            raw = raw + read * read_amp
            linear_camera = _demosaic_bilinear(raw.clamp(0.0, 1.0), bayer_pattern)
        else:
            linear_camera = linear * vignette_map
            linear_camera = linear_camera * (
                1.0 + prnu * float(prnu_strength) * preset["prnu"]
            )
            shot = _randn(linear_camera.shape, _safe_seed(seed, 17), device, dtype)
            read = _randn(linear_camera.shape, _safe_seed(seed, 29), device, dtype)
            shot_amp = float(shot_noise) * iso_scale * preset["shot"]
            read_amp = float(read_noise) * (iso_scale**1.15) * preset["read"]
            linear_camera = linear_camera + shot * torch.sqrt(
                linear_camera.clamp_min(0.0) + 1.0e-4
            ) * shot_amp
            linear_camera = linear_camera + read * read_amp

        linear_camera = _chromatic_aberration(
            linear_camera.clamp(0.0, 1.0), chromatic_aberration
        )
        out = _linear_to_srgb(linear_camera)

        if micro_contrast > 0.0:
            blur = _separable_blur(out, sigma=1.15)
            out = (out + (out - blur) * float(micro_contrast)).clamp(0.0, 1.0)

        luma = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        zone = _development_power(luma)
        curve = _prepare_development_curve(development_curve, out.shape, device, dtype)
        if curve is not None:
            zone = (zone * curve).clamp(0.0, 2.0)

        luma_grain = _multiscale_noise(
            b,
            1,
            h,
            w,
            _safe_seed(seed, 41),
            grain_size,
            device,
            dtype,
        )
        out = out + luma_grain * zone * float(grain_strength) * 0.040 * preset["grain"]

        if chroma_noise > 0.0:
            chroma = _multiscale_noise(
                b,
                3,
                h,
                w,
                _safe_seed(seed, 53),
                max(float(grain_size) * 1.6, 1.0),
                device,
                dtype,
            )
            chroma = chroma - chroma.mean(dim=1, keepdim=True)
            out = out + chroma * float(chroma_noise) * 0.035 * preset["chroma"]

        return out.clamp(0.0, 1.0)


class CameraISP_GrainSimulator:
    _engine: Optional[CameraISPEngine] = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (
                    ["iphone_12", "iphone_15", "full_frame", "neutral"],
                    {"default": "iphone_12"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1111111,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                        "control_after_generate": True,
                    },
                ),
                "effect_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "iso": (
                    "INT",
                    {"default": 400, "min": 50, "max": 6400, "step": 50},
                ),
                "bayer_pattern": (
                    ["RGGB", "BGGR", "GRBG", "GBRG", "off"],
                    {"default": "RGGB"},
                ),
                "prnu_strength": (
                    "FLOAT",
                    {"default": 0.008, "min": 0.0, "max": 0.05, "step": 0.001},
                ),
                "shot_noise": (
                    "FLOAT",
                    {"default": 0.010, "min": 0.0, "max": 0.08, "step": 0.001},
                ),
                "read_noise": (
                    "FLOAT",
                    {"default": 0.003, "min": 0.0, "max": 0.04, "step": 0.001},
                ),
                "grain_strength": (
                    "FLOAT",
                    {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "grain_size": (
                    "FLOAT",
                    {"default": 1.35, "min": 0.5, "max": 6.0, "step": 0.05},
                ),
                "chroma_noise": (
                    "FLOAT",
                    {"default": 0.006, "min": 0.0, "max": 0.08, "step": 0.001},
                ),
                "vignette": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.6, "step": 0.01},
                ),
                "chromatic_aberration": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "micro_contrast": (
                    "FLOAT",
                    {"default": 0.035, "min": 0.0, "max": 0.25, "step": 0.005},
                ),
                "prnu_tile_size": (
                    ["1024", "2048", "4096"],
                    {"default": "2048"},
                ),
            },
            "optional": {
                "profile_path": ("STRING", {"default": ""}),
                "development_curve": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "effect_delta")
    FUNCTION = "process"
    CATEGORY = "ImageProcessing/CameraISP"

    def process(
        self,
        image: torch.Tensor,
        profile: str,
        seed: int,
        effect_strength: float,
        iso: int,
        bayer_pattern: str,
        prnu_strength: float,
        shot_noise: float,
        read_noise: float,
        grain_strength: float,
        grain_size: float,
        chroma_noise: float,
        vignette: float,
        chromatic_aberration: float,
        micro_contrast: float,
        prnu_tile_size: str,
        profile_path: str = "",
        development_curve: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if self._engine is None:
            self._engine = CameraISPEngine()

        source = _to_bchw(image).to(dtype=torch.float32)
        strength = float(effect_strength)

        with torch.no_grad():
            processed = self._engine.apply(
                image=source,
                seed=seed,
                profile=profile,
                profile_path=profile_path,
                iso=iso,
                bayer_pattern=bayer_pattern,
                prnu_strength=prnu_strength,
                shot_noise=shot_noise,
                read_noise=read_noise,
                grain_strength=grain_strength,
                grain_size=grain_size,
                chroma_noise=chroma_noise,
                vignette=vignette,
                chromatic_aberration=chromatic_aberration,
                micro_contrast=micro_contrast,
                prnu_tile_size=int(prnu_tile_size),
                development_curve=development_curve,
            )
            output = torch.lerp(source, processed, strength).clamp(0.0, 1.0)

            mask_bchw = _prepare_mask(mask, source.shape, source.device, source.dtype)
            if mask_bchw is not None:
                output = source * (1.0 - mask_bchw) + output * mask_bchw

            delta = (output - source) * 6.0 + 0.5
            delta = delta.clamp(0.0, 1.0)

        return (_to_bhwc(output), _to_bhwc(delta))


NODE_CLASS_MAPPINGS = {
    "CameraISP_GrainSimulator": CameraISP_GrainSimulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraISP_GrainSimulator": "Camera ISP + Film Grain",
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = CameraISP_GrainSimulator()
    dummy = torch.rand(1, 512, 768, 3, device=device)
    result, delta = node.process(
        image=dummy,
        profile="iphone_12",
        seed=1111111,
        effect_strength=0.75,
        iso=400,
        bayer_pattern="RGGB",
        prnu_strength=0.008,
        shot_noise=0.010,
        read_noise=0.003,
        grain_strength=0.18,
        grain_size=1.35,
        chroma_noise=0.006,
        vignette=0.08,
        chromatic_aberration=0.20,
        micro_contrast=0.035,
        prnu_tile_size="1024",
    )
    print("Input:", tuple(dummy.shape))
    print("Output:", tuple(result.shape), "Delta:", tuple(delta.shape))
