"""Minimal example: predict BetaEarth embeddings from local GeoTIFFs.

Usage:
    python examples/predict.py \
        --s2_l2a path/to/s2_l2a_bands/ \
        --s1 path/to/s1_vv.tif path/to/s1_vh.tif \
        --dem path/to/dem.tif \
        --doy 182 \
        --output embedding.npy
"""

import argparse

import numpy as np


def load_s2_bands(band_dir: str) -> np.ndarray:
    """Load 9 S2 bands from a directory of GeoTIFFs.

    Expected files: B02.tif, B03.tif, B04.tif, B08.tif,
                    B05.tif, B06.tif, B07.tif, B11.tif, B12.tif
    Returns: (9, H, W) uint16
    """
    import rasterio
    from pathlib import Path

    band_dir = Path(band_dir)
    band_order = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12"]
    bands = []
    for name in band_order:
        with rasterio.open(band_dir / f"{name}.tif") as src:
            bands.append(src.read(1))
    return np.stack(bands, axis=0).astype(np.uint16)


def load_s1(vv_path: str, vh_path: str) -> np.ndarray:
    """Load S1 VV and VH from GeoTIFFs. Returns (2, H, W) float32."""
    import rasterio

    with rasterio.open(vv_path) as src:
        vv = src.read(1)
    with rasterio.open(vh_path) as src:
        vh = src.read(1)
    return np.stack([vv, vh], axis=0).astype(np.float32)


def load_dem(dem_path: str) -> np.ndarray:
    """Load DEM from GeoTIFF. Returns (1, H, W) float32."""
    import rasterio

    with rasterio.open(dem_path) as src:
        return src.read(1).astype(np.float32)[np.newaxis]


def main():
    parser = argparse.ArgumentParser(description="Predict BetaEarth embeddings")
    parser.add_argument("--model", default="asterisk-labs/betaearth-segformer-film",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--s2_l2a", help="Directory with S2 L2A band TIFFs")
    parser.add_argument("--s2_l1c", help="Directory with S2 L1C band TIFFs")
    parser.add_argument("--s1", nargs=2, metavar=("VV", "VH"),
                        help="Paths to S1 VV and VH GeoTIFFs")
    parser.add_argument("--dem", help="Path to DEM GeoTIFF")
    parser.add_argument("--doy", type=int, default=182, help="Day of year (1-366)")
    parser.add_argument("--output", default="embedding.npy", help="Output .npy path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    from betaearth import BetaEarth

    model = BetaEarth.from_pretrained(args.model, device=args.device)
    print(model)

    kwargs = {"doy": args.doy}

    if args.s2_l2a:
        kwargs["s2_l2a"] = load_s2_bands(args.s2_l2a)
        print(f"S2 L2A: {kwargs['s2_l2a'].shape}")

    if args.s2_l1c:
        kwargs["s2_l1c"] = load_s2_bands(args.s2_l1c)
        print(f"S2 L1C: {kwargs['s2_l1c'].shape}")

    if args.s1:
        kwargs["s1"] = load_s1(args.s1[0], args.s1[1])
        print(f"S1: {kwargs['s1'].shape}")

    if args.dem:
        kwargs["dem"] = load_dem(args.dem)
        print(f"DEM: {kwargs['dem'].shape}")

    embedding = model.predict(**kwargs)
    print(f"Embedding: {embedding.shape}, dtype={embedding.dtype}")

    np.save(args.output, embedding)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
