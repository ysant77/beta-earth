"""BetaEarth model for dense geospatial embedding prediction.

Standalone inference module — no training dependencies required.
Includes model architecture, preprocessing, and tiled inference.

Dependencies: torch, segmentation-models-pytorch, numpy
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import Tensor


# =============================================================================
# Constants
# =============================================================================

EMBEDDING_DIM = 64
TILE_SIZE_PX = 1068  # Major TOM grid: 10.68 km at 10m

S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12"]
S1_NODATA = -32768.0

# Zarr/raw band order → model band order (10m bands first, then 20m)
ZARR_TO_MODEL_BAND_ORDER = [0, 1, 2, 6, 3, 4, 5, 7, 8]


# =============================================================================
# Preprocessing
# =============================================================================

def normalise_s2(arr: np.ndarray, reorder_bands: bool = False) -> np.ndarray:
    """Normalise Sentinel-2 L1C or L2A from raw DN to [0, 1].

    Args:
        arr: (9, H, W) uint16 digital numbers (~0-10000).
        reorder_bands: If True, reorder from sequential band order
            [B02,B03,B04,B05,B06,B07,B08,B11,B12] to model order
            [B02,B03,B04,B08,B05,B06,B07,B11,B12].

    Returns:
        (9, H, W) float32 in ~[0, 1].
    """
    arr = arr.astype(np.float32)
    if reorder_bands:
        arr = arr[ZARR_TO_MODEL_BAND_ORDER]
    return arr / 10000.0


def normalise_s1(arr: np.ndarray) -> np.ndarray:
    """Normalise Sentinel-1 RTC from linear power to [0, 1].

    Args:
        arr: (2, H, W) float32 linear power (VV, VH), ~0-200.

    Returns:
        (2, H, W) float32 in [0, 1].
    """
    arr = arr.astype(np.float32)
    valid = (arr > 0) & (arr != S1_NODATA)
    arr = np.where(valid, arr, 1e-10)
    arr = 10.0 * np.log10(arr)  # dB
    arr = np.clip((arr + 25.0) / 25.0, 0.0, 1.0)
    return arr


def normalise_dem(arr: np.ndarray) -> np.ndarray:
    """Normalise COP-DEM from meters to [0, 1] via min-max scaling.

    Args:
        arr: (1, H, W) float32 elevation in meters.

    Returns:
        (1, H, W) float32 in [0, 1].
    """
    arr = arr.astype(np.float32)
    valid = arr > -30000
    arr = np.where(valid, arr, 0.0)
    vmin = float(arr[valid].min()) if valid.any() else 0.0
    vmax = float(arr[valid].max()) if valid.any() else 1.0
    denom = max(vmax - vmin, 1.0)
    return (arr - vmin) / denom


# =============================================================================
# Model components
# =============================================================================

class SinusoidalDOYEmbedding(nn.Module):
    """Encode day-of-year (1-366) as a fixed sinusoidal vector."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        freqs = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / embed_dim)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, doy: Tensor) -> Tensor:
        t = doy.float() * (2 * math.pi / 366.0)
        t = t.unsqueeze(-1)
        return torch.cat([torch.sin(t * self.freqs), torch.cos(t * self.freqs)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * features + beta."""

    def __init__(self, cond_dim: int, feature_channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:feature_channels] = 1.0

    def forward(self, features: Tensor, cond: Tensor) -> Tensor:
        gb = self.proj(cond)
        C = features.shape[1]
        gamma = gb[:, :C, None, None]
        beta = gb[:, C:, None, None]
        return gamma * features + beta


class TimestampConditioner(nn.Module):
    """Sinusoidal DOY embedding -> FiLM modulation."""

    def __init__(self, embed_dim: int = 128, feature_channels: int = 256):
        super().__init__()
        self.embedding = SinusoidalDOYEmbedding(embed_dim)
        self.film = FiLM(embed_dim, feature_channels)

    def forward(self, features: Tensor, doy: Tensor) -> Tensor:
        return self.film(features, self.embedding(doy))


class SegFormerEncoder(nn.Module):
    """SegFormer-B2 with per-modality encoders and FiLM time conditioning.

    Architecture:
        4 x MiT-B2 FPN encoders (S2-L1C, S2-L2A, S1-RTC, COP-DEM)
        FiLM temporal conditioning on S2 and S1 features
        Channel concatenation fusion -> 1x1 conv projection to 64-dim
    """

    def __init__(self, embed_dim: int = 64, encoder_name: str = "mit_b2",
                 fusion_dim: int = 256, time_embed_dim: int = 128):
        super().__init__()
        self.fusion_dim = fusion_dim

        self.s2l1c_encoder = smp.FPN(encoder_name=encoder_name, in_channels=9,
                                      classes=fusion_dim, encoder_weights="imagenet")
        self.s2l2a_encoder = smp.FPN(encoder_name=encoder_name, in_channels=9,
                                      classes=fusion_dim, encoder_weights="imagenet")
        self.s1_encoder = smp.FPN(encoder_name=encoder_name, in_channels=2,
                                   classes=fusion_dim, encoder_weights="imagenet")
        self.dem_encoder = smp.FPN(encoder_name=encoder_name, in_channels=1,
                                    classes=fusion_dim, encoder_weights="imagenet")

        self.time_cond = TimestampConditioner(time_embed_dim, fusion_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim * 4, fusion_dim, 1),
            nn.GELU(),
            nn.Conv2d(fusion_dim, embed_dim, 1),
        )

    def forward(self, batch: dict, modalities: list[str] | None = None) -> Tensor:
        if modalities is None:
            modalities = ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]

        target_size = None
        B = None
        for k in ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]:
            if k in batch and batch[k] is not None:
                target_size = (batch[k].shape[-2], batch[k].shape[-1])
                B = batch[k].shape[0]
                break

        doy = batch.get("timestamp")
        device = next(self.parameters()).device

        encoders = {
            "s2_l1c": self.s2l1c_encoder,
            "s2_l2a": self.s2l2a_encoder,
            "s1_rtc": self.s1_encoder,
            "cop_dem": self.dem_encoder,
        }
        time_conditioned = {"s2_l1c", "s2_l2a", "s1_rtc"}

        feats = []
        for mod_key in ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]:
            x = batch.get(mod_key)
            if mod_key in modalities and x is not None:
                feat = encoders[mod_key](x)
                if mod_key in time_conditioned and doy is not None:
                    feat = self.time_cond(feat, doy)
                feats.append(feat)
            else:
                feats.append(torch.zeros(B, self.fusion_dim, 1, 1, device=device))

        feats = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                 if f.shape[-2:] != target_size else f for f in feats]

        fused = torch.cat(feats, dim=1)
        out = self.fusion(fused)
        return out.permute(0, 2, 3, 1)  # (B, H, W, 64)


# =============================================================================
# Tiled inference
# =============================================================================

def _make_blend_window(tile_size: int, overlap: int) -> np.ndarray:
    """Create a 2D trapezoidal blending window."""
    w = np.ones(tile_size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.outer(w, w)  # (tile_size, tile_size)


def tiled_inference(model: nn.Module, batch: dict, tile_size: int = 224,
                    overlap: int = 32, modalities: list[str] | None = None) -> Tensor:
    """Run model on overlapping tiles with trapezoidal blending.

    Args:
        model: SegFormerEncoder (or compatible).
        batch: Dict with modality tensors, each (1, C, H, W) on GPU.
        tile_size: Tile edge length in pixels.
        overlap: Overlap between adjacent tiles.
        modalities: Optional modality subset.

    Returns:
        (1, H, W, 64) predicted embeddings.
    """
    # Get spatial dims from any available modality
    for k in ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]:
        if k in batch and batch[k] is not None:
            H, W = batch[k].shape[-2:]
            device = batch[k].device
            break

    stride = tile_size - overlap
    blend = torch.from_numpy(_make_blend_window(tile_size, overlap)).to(device)

    output = torch.zeros(1, H, W, EMBEDDING_DIM, device=device)
    weight = torch.zeros(1, H, W, 1, device=device)

    # Collect tile coordinates
    tiles = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y, H - tile_size)
            x1 = min(x, W - tile_size)
            tiles.append((y1, x1))

    # Deduplicate
    tiles = list(dict.fromkeys(tiles))

    # Process in chunks
    chunk_size = 12
    for i in range(0, len(tiles), chunk_size):
        chunk = tiles[i:i + chunk_size]

        tile_batches = []
        for y1, x1 in chunk:
            tile_batch = {}
            for k in ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]:
                if k in batch and batch[k] is not None:
                    tile_batch[k] = batch[k][:, :, y1:y1+tile_size, x1:x1+tile_size]
            if "timestamp" in batch:
                tile_batch["timestamp"] = batch["timestamp"]
            tile_batches.append(tile_batch)

        # Stack into a single batch
        stacked = {}
        for k in tile_batches[0]:
            if isinstance(tile_batches[0][k], Tensor):
                stacked[k] = torch.cat([tb[k] for tb in tile_batches], dim=0)
        if "timestamp" in batch:
            stacked["timestamp"] = batch["timestamp"].expand(len(chunk))

        preds = model(stacked, modalities=modalities)  # (N, ts, ts, 64)

        for j, (y1, x1) in enumerate(chunk):
            w = blend.unsqueeze(0).unsqueeze(-1)  # (1, ts, ts, 1)
            output[:, y1:y1+tile_size, x1:x1+tile_size] += preds[j:j+1] * w
            weight[:, y1:y1+tile_size, x1:x1+tile_size] += w

    return output / weight.clamp(min=1e-8)


# =============================================================================
# Main interface
# =============================================================================

class BetaEarth:
    """High-level interface for BetaEarth embedding prediction.

    Example::

        model = BetaEarth.from_pretrained("asterisk-labs/betaearth-segformer-film")
        emb = model.predict(s2_l2a=arr, dem=dem, doy=182)
    """

    def __init__(self, encoder: SegFormerEncoder, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, device: str = "cuda",
                        **kwargs) -> "BetaEarth":
        """Load a pretrained BetaEarth model.

        Args:
            repo_id_or_path: HuggingFace Hub repo ID (e.g. "asterisk-labs/betaearth-segformer-film")
                or local path to a .pt / .ckpt file.
            device: "cuda" or "cpu".

        Returns:
            BetaEarth instance ready for inference.
        """
        path = Path(repo_id_or_path)

        if path.exists() and path.is_file():
            # Local file
            weights_path = path
        elif path.exists() and path.is_dir():
            # Local directory — look for weights file
            weights_path = path / "model.pt"
            if not weights_path.exists():
                weights_path = path / "model.safetensors"
        else:
            # HuggingFace Hub
            from huggingface_hub import hf_hub_download
            try:
                weights_path = hf_hub_download(repo_id=repo_id_or_path,
                                                filename="model.pt")
            except Exception:
                weights_path = hf_hub_download(repo_id=repo_id_or_path,
                                                filename="model.safetensors")

        encoder = SegFormerEncoder(
            embed_dim=64, encoder_name="mit_b2",
            fusion_dim=256, time_embed_dim=128,
        )

        # Load weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state_dict, strict=False)

        return cls(encoder, device=device)

    @torch.no_grad()
    def predict(
        self,
        s2_l2a: np.ndarray | None = None,
        s2_l1c: np.ndarray | None = None,
        s1: np.ndarray | None = None,
        dem: np.ndarray | None = None,
        doy: int = 182,
        tile_size: int = 224,
        overlap: int = 32,
        normalise: bool = True,
        l2_norm: bool = True,
    ) -> np.ndarray:
        """Predict dense embeddings from satellite imagery.

        All inputs should be raw (unnormalised) values. Normalisation is
        handled internally unless ``normalise=False``.

        Args:
            s2_l2a: (9, H, W) Sentinel-2 L2A, uint16 DN (~0-10000).
                Band order: [B02, B03, B04, B08, B05, B06, B07, B11, B12].
            s2_l1c: (9, H, W) Sentinel-2 L1C, uint16 DN.
            s1: (2, H, W) Sentinel-1 RTC, float32 linear power (VV, VH).
            dem: (1, H, W) COP-DEM, float32 elevation in meters.
            doy: Day of year (1-366) for temporal conditioning.
            tile_size: Inference tile size (default 224).
            overlap: Tile overlap in pixels (default 32).
            normalise: Apply built-in normalisation (default True).
                Set False if inputs are already normalised to [0, 1].
            l2_norm: L2-normalise output per pixel (default True).

        Returns:
            (H, W, 64) float32 numpy array.
        """
        batch = {}
        modalities = []

        if s2_l1c is not None:
            arr = normalise_s2(s2_l1c) if normalise else s2_l1c.astype(np.float32)
            batch["s2_l1c"] = torch.from_numpy(arr)[None].to(self.device)
            modalities.append("s2_l1c")

        if s2_l2a is not None:
            arr = normalise_s2(s2_l2a) if normalise else s2_l2a.astype(np.float32)
            batch["s2_l2a"] = torch.from_numpy(arr)[None].to(self.device)
            modalities.append("s2_l2a")

        if s1 is not None:
            arr = normalise_s1(s1) if normalise else s1.astype(np.float32)
            batch["s1_rtc"] = torch.from_numpy(arr)[None].to(self.device)
            modalities.append("s1_rtc")

        if dem is not None:
            arr = normalise_dem(dem) if normalise else dem.astype(np.float32)
            batch["cop_dem"] = torch.from_numpy(arr)[None].to(self.device)
            modalities.append("cop_dem")

        if not modalities:
            raise ValueError("At least one input modality is required.")

        batch["timestamp"] = torch.tensor([doy], dtype=torch.long, device=self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            pred = tiled_inference(self.encoder, batch, tile_size=tile_size,
                                   overlap=overlap, modalities=modalities)

        emb = pred[0].cpu().float().numpy()  # (H, W, 64)

        if l2_norm:
            norms = np.linalg.norm(emb, axis=-1, keepdims=True)
            emb = emb / np.clip(norms, 1e-8, None)

        return emb

    def __repr__(self):
        n_params = sum(p.numel() for p in self.encoder.parameters())
        return f"BetaEarth(params={n_params/1e6:.1f}M, device={self.device})"
