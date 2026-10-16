"""Native PyTorch SCNet (Sparse Compression Network) vocal separator.

Architecture from:
"SCNet: Sparse Compression Network for Music Source Separation" (ICASSP 2024)
and ZFTurbo/Music-Source-Separation-Training (`SCNet-large_starrytong_fixed.ckpt`).
"""

import math
import os
import tempfile
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import is_available as is_cuda_available
from torch.nn.modules.rnn import LSTM

SCNET_LARGE_URL = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/"
    "releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt"
)

SCNET_LARGE_CONFIG = {
    "sources": ["drums", "bass", "other", "vocals"],
    "audio_channels": 2,
    "dims": [4, 64, 128, 256],
    "nfft": 4096,
    "hop_size": 1024,
    "win_size": 4096,
    "normalized": True,
    "band_SR": [0.225, 0.372, 0.403],
    "band_stride": [1, 4, 16],
    "band_kernel": [3, 4, 16],
    "conv_depths": [3, 2, 1],
    "compress": 4,
    "conv_kernel": 3,
    "num_dplayer": 6,
    "expand": 1,
}


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class FeatureConversion(nn.Module):
    def __init__(self, channels: int, inverse: bool):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inverse:
            x = x.float()
            x_r = x[:, : self.channels // 2, :, :]
            x_i = x[:, self.channels // 2 :, :, :]
            x_complex = torch.complex(x_r, x_i)
            return torch.fft.irfft(x_complex, dim=3, norm="ortho")
        x = x.float()
        x_complex = torch.fft.rfft(x, dim=3, norm="ortho")
        return torch.cat([x_complex.real, x_complex.imag], dim=1)


class DualPathRNN(nn.Module):
    def __init__(self, d_model: int, expand: int, bidirectional: bool = True):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        self.lstm_layers = nn.ModuleList(
            [LSTM(d_model, self.hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True) for _ in range(2)]
        )
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, d_model) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        original_x = x
        x = self.norm_layers[0](x)
        x = x.transpose(1, 3).contiguous().view(b * t, f, c)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(b, t, f, c).transpose(1, 3)
        x = x + original_x

        original_x = x
        x = self.norm_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(b * f, c, t).transpose(1, 2)
        x, _ = self.lstm_layers[1](x)
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(b, f, c, t).transpose(1, 2)
        return x + original_x


class SeparationNet(nn.Module):
    def __init__(self, channels: int, expand: int = 1, num_layers: int = 6):
        super().__init__()
        self.num_layers = num_layers
        self.dp_modules = nn.ModuleList(
            [DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand) for i in range(num_layers)]
        )
        self.feature_conversion = nn.ModuleList(
            [FeatureConversion(channels * 2, inverse=(i % 2 == 1)) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversion[i](x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, depth: int = 2, compress: int = 4, kernel: int = 3):
        super().__init__()
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            padding = kernel // 2
            layer = nn.Sequential(
                nn.GroupNorm(1, channels),
                nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                nn.GLU(1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                nn.GroupNorm(1, hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = x + skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        return F.glu(x, dim=1)


class SDlayer(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, band_configs: Dict[str, Dict[str, float]]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.strides: List[int] = []
        self.kernels: List[int] = []
        for config in band_configs.values():
            kernel = int(config["kernel"])
            stride = int(config["stride"])
            self.convs.append(nn.Conv2d(channels_in, channels_out, (kernel, 1), (stride, 1), (0, 0)))
            self.strides.append(stride)
            self.kernels.append(kernel)
        self.SR_low = float(band_configs["low"]["SR"])
        self.SR_mid = float(band_configs["mid"]["SR"])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        fr = x.shape[2]
        splits = [
            (0, math.ceil(fr * self.SR_low)),
            (math.ceil(fr * self.SR_low), math.ceil(fr * (self.SR_low + self.SR_mid))),
            (math.ceil(fr * (self.SR_low + self.SR_mid)), fr),
        ]
        outputs: List[torch.Tensor] = []
        original_lengths: List[int] = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]
            total_padding = (kernel - stride) if stride == 1 else ((stride - current_length % stride) % stride)
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left
            padded = F.pad(extracted, (0, 0, pad_left, pad_right))
            outputs.append(conv(padded))
        return outputs, original_lengths


class SUlayer(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, band_configs: Dict[str, Dict[str, float]]):
        super().__init__()
        self.convtrs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels_in, channels_out, (int(cfg["kernel"]), 1), (int(cfg["stride"]), 1))
                for cfg in band_configs.values()
            ]
        )

    def forward(self, x: torch.Tensor, lengths: List[int], origin_lengths: List[int]) -> torch.Tensor:
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None),
        ]
        outputs: List[torch.Tensor] = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            current_fr_length = out.shape[2]
            dist = abs(origin_lengths[idx] - current_fr_length) // 2
            outputs.append(out[:, :, dist : dist + origin_lengths[idx], :])
        return torch.cat(outputs, dim=2)


class SDblock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        band_configs: Dict[str, Dict[str, float]],
        conv_config: Dict[str, int],
        depths: List[int],
        kernel_size: int = 3,
    ):
        super().__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList([ConvolutionModule(channels_out, depth, **conv_config) for depth in depths])
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        bands, original_lengths = self.SDlayer(x)
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)
        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        return self.globalconv(full_band), full_band, lengths, original_lengths


class SCNet(nn.Module):
    def __init__(
        self,
        sources: List[str],
        audio_channels: int = 2,
        dims: List[int] = [4, 64, 128, 256],
        nfft: int = 4096,
        hop_size: int = 1024,
        win_size: int = 4096,
        normalized: bool = True,
        band_SR: List[float] = [0.225, 0.372, 0.403],
        band_stride: List[int] = [1, 4, 16],
        band_kernel: List[int] = [3, 4, 16],
        conv_depths: List[int] = [3, 2, 1],
        compress: int = 4,
        conv_kernel: int = 3,
        num_dplayer: int = 6,
        expand: int = 1,
    ):
        super().__init__()
        self.sources = list(sources)
        self.audio_channels = audio_channels
        self.dims = list(dims)
        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {"SR": band_SR[i], "stride": band_stride[i], "kernel": band_kernel[i]}
            for i in range(len(band_keys))
        }
        self.hop_length = hop_size
        self.conv_config = {"compress": compress, "kernel": conv_kernel}
        self.stft_config = {
            "n_fft": nfft,
            "hop_length": hop_size,
            "win_length": win_size,
            "center": True,
            "normalized": normalized,
        }
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            self.encoder.append(
                SDblock(
                    channels_in=dims[index],
                    channels_out=dims[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=conv_depths,
                )
            )
            self.decoder.insert(
                0,
                nn.Sequential(
                    FusionLayer(channels=dims[index + 1]),
                    SUlayer(
                        channels_in=dims[index + 1],
                        channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                        band_configs=self.band_configs,
                    ),
                ),
            )

        self.separation_net = SeparationNet(channels=dims[-1], expand=expand, num_layers=num_dplayer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        x = F.pad(x, (0, padding))

        length = x.shape[-1]
        x = x.reshape(-1, length)
        x = torch.stft(x, **self.stft_config, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels, x.shape[1], x.shape[2]
        )

        b, _c, fr, t = x.shape
        save_skip: deque[torch.Tensor] = deque()
        save_lengths: deque[List[int]] = deque()
        save_original_lengths: deque[List[int]] = deque()

        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        x = self.separation_net(x)

        for dec_stage in self.decoder:
            fusion_layer, su_layer = list(dec_stage.children())
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        n = self.dims[0]
        x = x.view(b, n, -1, fr, t)
        x = x.reshape(-1, 2, fr, t).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self.stft_config)
        x = x.reshape(b, len(self.sources), self.audio_channels, -1)
        return x[:, :, :, :-padding]


def _windowing_array(chunk_size: int, fade_size: int, device: torch.device) -> torch.Tensor:
    fadein = torch.linspace(0.0, 1.0, fade_size, device=device)
    fadeout = torch.linspace(1.0, 0.0, fade_size, device=device)
    window = torch.ones(chunk_size, device=device)
    window[:fade_size] = fadein
    window[-fade_size:] = fadeout
    return window


def _download_checkpoint(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        return
    with tempfile.NamedTemporaryFile(dir=str(target_path.parent), delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        os.replace(tmp_path, target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class SCNetSeparatorAdapter:
    """Native PyTorch SCNet-Large separator with the same separate_tensor contract as Demucs."""

    samplerate = 44100
    audio_channels = 2
    chunk_size = 485100  # 11s * 44100
    num_overlap = 4

    def __init__(self, model_filename: str = "SCNet-large_starrytong_fixed.ckpt", model_dir: Optional[str] = None):
        cache_dir = Path(model_dir or "./_model_cache")
        ckpt_path = cache_dir / model_filename
        if not ckpt_path.exists():
            _download_checkpoint(SCNET_LARGE_URL, ckpt_path)

        device_name = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model = SCNet(**SCNET_LARGE_CONFIG)  # type: ignore[arg-type]

        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("state", "state_dict", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def separate_tensor(self, wav: torch.Tensor, sr: Optional[int] = None):
        mix_orig = wav.detach().cpu().float()
        mono = mix_orig.mean(0)
        mean = mono.mean()
        std = mono.std().clamp_min(1e-8)
        mix = (mix_orig - mean) / std

        chunk_size = self.chunk_size
        step = chunk_size // self.num_overlap
        fade_size = chunk_size // 10
        border = chunk_size - step
        length_init = mix.shape[-1]

        if length_init > 2 * border and border > 0:
            mix = F.pad(mix, (border, border), mode="reflect")

        num_sources = len(self.model.sources)
        result = torch.zeros((num_sources, *mix.shape), dtype=torch.float32)
        counter = torch.zeros(mix.shape[-1], dtype=torch.float32)
        base_window = _windowing_array(chunk_size, fade_size, torch.device("cpu"))

        with torch.inference_mode():
            i = 0
            total_len = mix.shape[-1]
            while i < total_len:
                part = mix[:, i : i + chunk_size]
                seg_len = part.shape[-1]
                pad_mode = "reflect" if seg_len > chunk_size // 2 else "constant"
                if seg_len < chunk_size:
                    part = F.pad(part, (0, chunk_size - seg_len), mode=pad_mode, value=0.0)

                pred = self.model(part.unsqueeze(0).to(self.device))[0].cpu().float()

                window = base_window.clone()
                if i == 0:
                    window[:fade_size] = 1.0
                if i + step >= total_len:
                    window[-fade_size:] = 1.0

                w_seg = window[:seg_len]
                result[..., i : i + seg_len] += pred[..., :seg_len] * w_seg
                counter[i : i + seg_len] += w_seg
                i += step

        estimated = result / counter.clamp_min(1e-8)
        if length_init > 2 * border and border > 0:
            estimated = estimated[..., border:-border]
        estimated = estimated[..., :length_init] * std + mean

        outputs = {source: estimated[idx] for idx, source in enumerate(self.model.sources)}
        return wav, outputs
