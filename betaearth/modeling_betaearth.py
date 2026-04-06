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
# DINOv3 primitive-based encoder
# =============================================================================

_DPT_LAYERS_S = [2, 5, 8, 11]    # ViT-S/16: 12 blocks
_DPT_LAYERS_L = [5, 11, 17, 23]  # ViT-L/16: 24 blocks

# Band indices within 9-band S2 stack: [B02,B03,B04,B08,B05,B06,B07,B11,B12]
_B02, _B03, _B04, _B08 = 0, 1, 2, 3
_B05, _B06, _B07, _B11, _B12 = 4, 5, 6, 7, 8


def _s2_to_primitives(x: Tensor) -> list[Tensor]:
    """(B, 9, H, W) → 4 × (B, 3, H, W): RGB, FalseIR, SWIR, RedEdge."""
    return [
        x[:, [_B04, _B03, _B02]],
        x[:, [_B08, _B04, _B03]],
        x[:, [_B12, _B11, _B04]],
        x[:, [_B07, _B06, _B05]],
    ]


def _s1_to_primitives(x: Tensor) -> list[Tensor]:
    """(B, 2, H, W) → 1 × (B, 3, H, W): VV, VH, VV/VH."""
    vv, vh = x[:, 0:1], x[:, 1:2]
    ratio = (vv / (vh + 1e-6)).clamp(0, 1)
    return [torch.cat([vv, vh, ratio], dim=1)]


def _dem_to_primitives(x: Tensor) -> list[Tensor]:
    """(B, 1, H, W) → 1 × (B, 3, H, W): elevation, slope, aspect."""
    elev = x[:, 0:1]
    padded = F.pad(elev, (1, 1, 1, 1), mode="replicate")
    dz_dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
    dz_dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
    slope = torch.sqrt(dz_dx ** 2 + dz_dy ** 2).clamp(0, 1)
    aspect = (torch.atan2(dz_dy, dz_dx) + torch.pi) / (2 * torch.pi)
    return [torch.cat([elev, slope, aspect], dim=1)]


class DPTHead(nn.Module):
    """DPT-style reassembly: 4 ViT layers → dense feature map at 1/4 res."""

    def __init__(self, embed_dim: int, head_dim: int = 128):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(embed_dim, head_dim, 1) for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(head_dim * 4, head_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_dim, head_dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, layer_features: list[Tensor]) -> Tensor:
        target_h = layer_features[0].shape[2] * 4
        target_w = layer_features[0].shape[3] * 4
        projected = []
        for feat, proj in zip(layer_features, self.projections):
            p = F.interpolate(proj(feat), size=(target_h, target_w),
                              mode="bilinear", align_corners=False)
            projected.append(p)
        return self.fuse(torch.cat(projected, dim=1))


class SetFusion(nn.Module):
    """Set-based fusion: self-attention over primitives + learned query pooling."""

    def __init__(self, feat_dim: int = 128, out_dim: int = 256,
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.out_dim = out_dim
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, n_heads, dropout=0.0, batch_first=True,
        )
        self.cross_norm_q = nn.LayerNorm(feat_dim)
        self.cross_norm_kv = nn.LayerNorm(feat_dim)
        self.proj = nn.Sequential(nn.Linear(feat_dim, out_dim), nn.GELU())

    def forward(self, primitives: list[Tensor]) -> Tensor:
        B, C, H, W = primitives[0].shape
        N = len(primitives)
        x = torch.stack(primitives, dim=-1).permute(0, 2, 3, 4, 1).reshape(B * H * W, N, C)
        x = self.self_attn(x)
        q = self.cross_norm_q(self.query.expand(B * H * W, -1, -1))
        out, _ = self.cross_attn(q, self.cross_norm_kv(x), self.cross_norm_kv(x))
        out = self.proj(out.squeeze(1))
        return out.reshape(B, H, W, self.out_dim).permute(0, 3, 1, 2)


class EmbeddingDecoder(nn.Module):
    """Project fused features to 64-d embedding space."""

    def __init__(self, in_channels: int = 256, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, embed_dim, 1),
        )

    def forward(self, x: Tensor, target_size: tuple[int, int] | None = None) -> Tensor:
        out = self.proj(x)
        if target_size is not None and (out.shape[2], out.shape[3]) != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear",
                                align_corners=False)
        return out.permute(0, 2, 3, 1)


class DINOv3Encoder(nn.Module):
    """DINOv3 primitive-based encoder with set fusion.

    Architecture:
        Modalities → 3-band primitives → shared frozen DINOv3 + DPT head
        → FiLM temporal conditioning → SetFusion → EmbeddingDecoder → (B, H, W, 64)

    The DINOv3 backbone is loaded from torch.hub at runtime.
    Only the DPT head, FiLM, fusion, and decoder weights are stored.
    """

    VARIANTS = {
        "vits16": {"embed_dim": 384, "hub_name": "dinov3_vits16", "dpt_layers": _DPT_LAYERS_S},
        "vitl16": {"embed_dim": 1024, "hub_name": "dinov3_vitl16", "dpt_layers": _DPT_LAYERS_L},
    }

    def __init__(self, variant: str = "vitl16", head_dim: int = 128,
                 fusion_dim: int = 256, embed_dim: int = 64,
                 time_embed_dim: int = 128, backbone: nn.Module | None = None):
        super().__init__()
        cfg = self.VARIANTS[variant]
        self.variant = variant
        self.head_dim = head_dim
        self.dpt_layers = cfg["dpt_layers"]

        # Backbone — injected or loaded from hub
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = None  # Loaded lazily in load_backbone()

        # DPT head
        self.dpt_head = DPTHead(cfg["embed_dim"], head_dim)
        # FiLM time conditioning
        self.time_cond = TimestampConditioner(time_embed_dim, head_dim)
        # Set fusion
        self.fusion = SetFusion(head_dim, fusion_dim)
        # Decoder
        self.decoder = EmbeddingDecoder(fusion_dim, embed_dim)

    def load_backbone(self, weights_path: str | None = None):
        """Load frozen DINOv3 backbone from torch.hub or local weights."""
        cfg = self.VARIANTS[self.variant]
        if weights_path:
            import pathlib
            hub_dir = str(pathlib.Path(weights_path).parent)
            self.backbone = torch.hub.load(
                hub_dir, cfg["hub_name"], source="local", weights=weights_path,
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov3", cfg["hub_name"],
            )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _extract_features(self, x: Tensor) -> list[Tensor]:
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                x, n=len(self.backbone.blocks), reshape=True,
            )
        return [features[i].detach() for i in self.dpt_layers]

    def _encode_primitive(self, x: Tensor) -> Tensor:
        return self.dpt_head(self._extract_features(x))

    def forward(self, batch: dict, modalities: list[str] | None = None) -> Tensor:
        if modalities is None:
            modalities = ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]

        doy = batch.get("timestamp")
        target_size = None
        for k in ["s2_l1c", "s2_l2a", "s1_rtc", "cop_dem"]:
            if k in batch and batch[k] is not None:
                target_size = (batch[k].shape[-2], batch[k].shape[-1])
                break

        prim_fns = {
            "s2_l1c": (_s2_to_primitives, True),
            "s2_l2a": (_s2_to_primitives, True),
            "s1_rtc": (_s1_to_primitives, True),
            "cop_dem": (_dem_to_primitives, False),
        }

        all_feats = []
        for mod_key in modalities:
            x = batch.get(mod_key)
            if x is None:
                continue
            prim_fn, uses_time = prim_fns[mod_key]
            for prim in prim_fn(x):
                feat = self._encode_primitive(prim)
                if uses_time and doy is not None:
                    feat = self.time_cond(feat, doy)
                all_feats.append(feat)

        fused = self.fusion(all_feats)
        return self.decoder(fused, target_size)


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

    # Map repo names to model types
    _DINOV3_REPOS = {"betaearth-dinov3-vitl16", "betaearth-dinov3-vits16"}

    def __init__(self, encoder: nn.Module, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, device: str = "cuda",
                        dinov3_weights: str | None = None,
                        **kwargs) -> "BetaEarth":
        """Load a pretrained BetaEarth model.

        Args:
            repo_id_or_path: HuggingFace Hub repo ID (e.g. "asterisk-labs/betaearth-segformer-film"
                or "asterisk-labs/betaearth-dinov3-vitl16") or local path.
            device: "cuda" or "cpu".
            dinov3_weights: Path to DINOv3 backbone weights (.pth). Only needed
                for DINOv3 models. If None, downloads from torch.hub.

        Returns:
            BetaEarth instance ready for inference.
        """
        path = Path(repo_id_or_path)

        # Resolve weights path
        if path.exists() and path.is_file():
            weights_path = path
            config_path = None
        elif path.exists() and path.is_dir():
            weights_path = path / "model.pt"
            if not weights_path.exists():
                weights_path = path / "model.safetensors"
            config_path = path / "config.json" if (path / "config.json").exists() else None
        else:
            from huggingface_hub import hf_hub_download
            try:
                weights_path = hf_hub_download(repo_id=repo_id_or_path,
                                                filename="model.pt")
            except Exception:
                weights_path = hf_hub_download(repo_id=repo_id_or_path,
                                                filename="model.safetensors")
            try:
                config_path = hf_hub_download(repo_id=repo_id_or_path,
                                               filename="config.json")
            except Exception:
                config_path = None

        # Detect model type from config or repo name
        is_dinov3 = False
        variant = "vitl16"
        if config_path:
            import json
            with open(config_path) as f:
                config = json.load(f)
            is_dinov3 = config.get("model_type", "").startswith("betaearth-dinov3")
            variant = config.get("architecture", {}).get("variant", "vitl16")
        else:
            repo_name = Path(repo_id_or_path).name
            if any(d in repo_name for d in cls._DINOV3_REPOS):
                is_dinov3 = True
                variant = "vits16" if "vits16" in repo_name else "vitl16"

        if is_dinov3:
            encoder = DINOv3Encoder(
                variant=variant, head_dim=128,
                fusion_dim=256, embed_dim=64, time_embed_dim=128,
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict, strict=False)
            # Load frozen backbone
            encoder.load_backbone(dinov3_weights)
        else:
            encoder = SegFormerEncoder(
                embed_dim=64, encoder_name="mit_b2",
                fusion_dim=256, time_embed_dim=128,
            )
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
