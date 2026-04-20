#!/usr/bin/env python3
"""Generate BetaEarth embeddings for any area of interest.

Downloads Sentinel-2 L2A, Sentinel-1 RTC, and COP-DEM from Microsoft
Planetary Computer, runs inference, and writes per-timestamp + annual
average embeddings as Cloud-Optimised GeoTIFFs.

Requirements:
    pip install betaearth[rasterio] pystac-client planetary-computer \
        rioxarray scikit-learn

Usage:
    # By bounding box
    betaearth-generate --bbox 13.18 48.86 13.65 49.13 --years 2023

    # By OSM relation (e.g. Bavarian Forest National Park)
    betaearth-generate --osm_relation 1864214 --years 2023

    # Keep raw scenes + use any scene with >= 80% coverage
    betaearth-generate --bbox 13.18 48.86 13.65 49.13 \
        --years 2023 --min_coverage 80 --save_scenes

Output:
    outputs/{name}/
      2023.tif                      64-band COG — annual average embedding
      2023_preview_pca.png           3-band PCA-RGB preview (for quick browsing)
      2023_manifest.json             Metadata + list of scenes used
      2023_files/
        2023-01-01_s1/
          embedding.tif              64-band COG — this timestamp
          preview_pca.png            PCA-RGB preview (shared colour basis)
          scene.tif                  Raw input (only with --save_scenes)
        2023-02-07_s2_33UUQ/
          embedding.tif
          preview_pca.png
          scene.tif
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("betaearth.generate")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PC_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
RESOLUTION = 10.0  # metres


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def _elapsed(t0: float) -> str:
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


# ---------------------------------------------------------------------------
# AOI + grid
# ---------------------------------------------------------------------------
def resolve_aoi(args) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326."""
    if args.osm_relation:
        from shapely.geometry import shape as shapely_shape
        from urllib.request import urlopen

        url = f"https://polygons.openstreetmap.fr/get_geojson.py?id={args.osm_relation}&params=0"
        log.info("Fetching OSM relation %d ...", args.osm_relation)
        geojson = json.loads(urlopen(url).read())
        geom = shapely_shape(geojson)
        w, s, e, n = geom.bounds
        pad_lat, pad_lon = 0.009, 0.013
        bbox = (w - pad_lon, s - pad_lat, e + pad_lon, n + pad_lat)
        log.info("  Boundary: (%.3f, %.3f, %.3f, %.3f)", w, s, e, n)
        log.info("  Padded:   (%.3f, %.3f, %.3f, %.3f)", *bbox)
        return bbox
    if args.bbox:
        return tuple(args.bbox)
    if args.geojson:
        from shapely.geometry import shape as shapely_shape

        with open(args.geojson) as f:
            geojson = json.load(f)
        geom = geojson if geojson["type"] != "FeatureCollection" else geojson["features"][0]["geometry"]
        return shapely_shape(geom).bounds
    raise ValueError("Provide --bbox, --geojson, or --osm_relation")


def compute_grid(bbox_4326):
    """Compute UTM grid covering the bounding box at 10m resolution."""
    from pyproj import CRS, Transformer
    import rasterio.transform

    w, s, e, n = bbox_4326
    if w > e:
        raise ValueError(
            f"Bounding box crosses the antimeridian (west={w} > east={e}). "
            "Not supported — please split into two bboxes on either side of ±180°."
        )
    if n <= s:
        raise ValueError(f"Invalid bounding box: north ({n}) must be > south ({s}).")
    if abs(s) > 84 or abs(n) > 84:
        raise ValueError(
            f"Bounding box exceeds UTM coverage (|lat| > 84°). Use UPS instead."
        )

    lon_mid = (w + e) / 2
    utm_zone = int((lon_mid + 180) / 6) + 1
    hemisphere = "north" if (s + n) / 2 >= 0 else "south"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    crs = CRS.from_epsg(epsg)

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = to_utm.transform(bbox_4326[0], bbox_4326[1])
    x1, y1 = to_utm.transform(bbox_4326[2], bbox_4326[3])
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    # Snap to 10m grid
    x0 = np.floor(x0 / RESOLUTION) * RESOLUTION
    y1_snap = np.ceil(y1 / RESOLUTION) * RESOLUTION
    w = int(np.ceil((x1 - x0) / RESOLUTION))
    h = int(np.ceil((y1_snap - y0) / RESOLUTION))

    transform = rasterio.transform.from_origin(x0, y1_snap, RESOLUTION, RESOLUTION)
    return {
        "bbox_4326": bbox_4326,
        "epsg": epsg,
        "crs": crs,
        "transform": transform,
        "bounds": (x0, y0, x1, y1),
        "shape": (h, w),
    }


# ---------------------------------------------------------------------------
# Data download (Planetary Computer)
# ---------------------------------------------------------------------------
def _search_stac(bbox, year, collection, max_cloud=None, max_items=200):
    """Search Planetary Computer STAC for items.

    Note: callers are encouraged to pass max_cloud=None and apply cloud filtering
    in _seasonal_select instead — this preserves seasonal balance in temperate
    zones where summer is cloudier than winter (so a hard cloud filter ends up
    heavily winter-biasing the annual mosaic)."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(PC_CATALOG, modifier=planetary_computer.sign_inplace)
    query = {}
    if max_cloud is not None:
        query["eo:cloud_cover"] = {"lt": max_cloud}
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{year}-01-01/{year}-12-31",
        query=query,
        max_items=max_items,
    )
    return list(search.items())


def _seasonal_select(items, max_per_quarter=6, use_cloud=True, max_cloud=None):
    """Select up to max_per_quarter items per quarter.

    If max_cloud is given, scenes under that threshold are strongly preferred,
    but if a quarter has *no* scenes under the threshold we fall back to its
    least-cloudy scene anyway — preserving seasonal balance over strict cloud
    filtering (a common failure mode in temperate/tropical regions where summer
    is the cloudiest quarter).
    """
    quarters = {1: [], 2: [], 3: [], 4: []}
    for item in items:
        q = (item.datetime.month - 1) // 3 + 1
        quarters[q].append(item)

    selected = []
    for q in range(1, 5):
        pool = quarters[q]
        if not pool:
            continue
        if use_cloud:
            pool = sorted(pool, key=lambda x: x.properties.get("eo:cloud_cover", 100))
            if max_cloud is not None:
                under = [x for x in pool if x.properties.get("eo:cloud_cover", 100) < max_cloud]
                pool = under if under else pool[:1]  # seasonal-balance fallback
        selected.extend(pool[:max_per_quarter])
    return selected


def download_s2_cloud_mask(item, grid):
    """Fetch the Sentinel-2 Scene Classification Layer and return a bool mask
    of *usable* pixels (True = keep). Classes rejected: 0 nodata, 1 saturated,
    3 shadow, 8 cloud-med, 9 cloud-high, 10 thin cirrus. Snow/ice (11) is kept
    because for many AOIs it IS the winter land cover."""
    asset = item.assets.get("SCL") or item.assets.get("scl")
    if asset is None:
        return None
    scl = download_reprojected(asset.href, grid).astype(np.uint8)
    bad = np.isin(scl, (0, 1, 3, 8, 9, 10))
    return ~bad


def download_reprojected(url, grid):
    """Download a raster asset and reproject to the AOI grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    h, w = grid["shape"]
    with rasterio.open(url) as src:
        out = np.zeros((h, w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.bilinear,
        )
    return out


def download_s2(item, grid):
    """Download S2 L2A scene → (9, H, W) uint16."""
    band_map = {
        "B02": "B02", "B03": "B03", "B04": "B04", "B08": "B08",
        "B05": "B05", "B06": "B06", "B07": "B07", "B11": "B11", "B12": "B12",
    }
    h, w = grid["shape"]
    bands = np.zeros((9, h, w), dtype=np.float32)
    for i, (name, asset_key) in enumerate(band_map.items()):
        url = item.assets[asset_key].href
        bands[i] = download_reprojected(url, grid)
    return bands.astype(np.uint16)


def download_s1(item, grid):
    """Download S1 RTC scene → (2, H, W) float32."""
    h, w = grid["shape"]
    bands = np.zeros((2, h, w), dtype=np.float32)
    bands[0] = download_reprojected(item.assets["vv"].href, grid)
    bands[1] = download_reprojected(item.assets["vh"].href, grid)
    return bands


def download_dem(grid):
    """Download COP-DEM 30m → (1, H, W) float32."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(PC_CATALOG, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["cop-dem-glo-30"], bbox=list(grid["bbox_4326"]), max_items=20,
    )
    items = list(search.items())
    if not items:
        return None

    h, w = grid["shape"]
    dem_sum = np.zeros((h, w), dtype=np.float64)
    dem_count = np.zeros((h, w), dtype=np.float32)
    for item in items:
        tile = download_reprojected(item.assets["data"].href, grid)
        valid = tile != 0
        dem_sum[valid] += tile[valid]
        dem_count[valid] += 1
    dem_count = np.clip(dem_count, 1, None)
    return (dem_sum / dem_count).astype(np.float32)[np.newaxis]


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------
def check_coverage(data: np.ndarray) -> float:
    """Percent of AOI pixels that have valid (non-zero) data."""
    if data.ndim == 3:
        valid = np.any(data != 0, axis=0)
    else:
        valid = data != 0
    return 100.0 * valid.sum() / valid.size


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------
def write_geotiff(arr, grid, path, band_first=False):
    """Write array to a Cloud-Optimised GeoTIFF.

    - ZSTD compression with floating-point predictor (falls back to DEFLATE)
    - 256x256 tiles, pixel interleave
    """
    import rasterio

    if band_first:
        c, h, w = arr.shape
    else:
        h, w, c = arr.shape

    dtype = "uint16" if arr.dtype == np.uint16 else "float32"
    predictor = 2 if dtype == "uint16" else 3

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff", "dtype": dtype,
        "width": w, "height": h, "count": c,
        "crs": grid["crs"], "transform": grid["transform"],
        "compress": "zstd", "zstd_level": 9, "predictor": predictor,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "interleave": "pixel",
        # Auto-promote to BigTIFF when the raw payload would exceed 4 GB
        # (standard TIFF offsets are 32-bit). Covers large AOIs / 64-band outputs.
        "bigtiff": "IF_SAFER",
    }
    try:
        with rasterio.open(path, "w", **profile) as dst:
            for b in range(c):
                band = arr[b] if band_first else arr[:, :, b]
                dst.write(np.ascontiguousarray(band), b + 1)
    except Exception:
        profile["compress"] = "deflate"
        profile["zlevel"] = 9
        profile.pop("zstd_level", None)
        with rasterio.open(path, "w", **profile) as dst:
            for b in range(c):
                band = arr[b] if band_first else arr[:, :, b]
                dst.write(np.ascontiguousarray(band), b + 1)


# ---------------------------------------------------------------------------
# PCA preview
# ---------------------------------------------------------------------------
def fit_pca(emb):
    """Fit PCA on (H, W, 64) embedding, return reusable state."""
    from sklearn.decomposition import PCA

    flat = emb.reshape(-1, 64)
    valid = np.linalg.norm(flat, axis=1) > 1e-6
    pca = PCA(n_components=3)
    pca.fit(flat[valid])
    proj = pca.transform(flat)
    lo = np.array([np.percentile(proj[valid, c], 2) for c in range(3)])
    hi = np.array([np.percentile(proj[valid, c], 98) for c in range(3)])
    return {"pca": pca, "lo": lo, "hi": hi}


def write_pca_preview(emb, path, pca_state=None):
    """Write 3-band uint8 PCA-RGB preview as PNG."""
    from PIL import Image
    from sklearn.decomposition import PCA

    h, w = emb.shape[:2]
    flat = emb.reshape(-1, 64)
    valid = np.linalg.norm(flat, axis=1) > 1e-6

    if pca_state is not None:
        pca, lo, hi = pca_state["pca"], pca_state["lo"], pca_state["hi"]
    else:
        pca = PCA(n_components=3)
        pca.fit(flat[valid])
        proj = pca.transform(flat)
        lo = np.array([np.percentile(proj[valid, c], 2) for c in range(3)])
        hi = np.array([np.percentile(proj[valid, c], 98) for c in range(3)])

    proj = pca.transform(flat)
    rgb = np.zeros((flat.shape[0], 3), dtype=np.float32)
    for c in range(3):
        rgb[:, c] = (proj[:, c] - lo[c]) / (hi[c] - lo[c])
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rgb[~valid] = 0
    rgb = rgb.reshape(h, w, 3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def generate(
    bbox_4326,
    year: int,
    model,
    grid: dict,
    output_dir: Path,
    max_cloud: int = 20,
    max_per_quarter: int = 6,
    min_coverage: float = 100.0,
    overlap: int = 112,
    save_scenes: bool = False,
    save_per_timestamp_embedding: bool = True,
):
    """Full pipeline: search -> download -> predict -> write."""

    # --- Search ---
    # We deliberately don't filter by cloud at search time. The cloud filter
    # is applied in _seasonal_select with a per-quarter fallback so the annual
    # mosaic stays seasonally balanced even if summer is mostly cloudy.
    log.info("Searching Planetary Computer for %d ...", year)
    t0 = time.time()

    s2_all = _search_stac(bbox_4326, year, "sentinel-2-l2a")
    s2_items = _seasonal_select(s2_all, max_per_quarter=max_per_quarter, max_cloud=max_cloud)
    s1_all = _search_stac(bbox_4326, year, "sentinel-1-rtc")
    s1_items = _seasonal_select(s1_all, max_per_quarter=1, use_cloud=False)

    log.info("  Found %d S2 L2A + %d S1 RTC scenes (%s)", len(s2_items), len(s1_items), _elapsed(t0))

    # --- DEM (download once, cache) ---
    dem_path = output_dir / "dem_cache.npy"
    if dem_path.exists():
        log.info("Loading cached DEM ...")
        dem = np.load(dem_path)
    else:
        log.info("Downloading COP-DEM 30m ...")
        t0 = time.time()
        dem = download_dem(grid)
        if dem is not None:
            np.save(dem_path, dem)
            log.info("  DEM: range [%.0f, %.0f]m (%s)", dem.min(), dem.max(), _elapsed(t0))
        else:
            log.warning("  DEM download failed — continuing without DEM")

    # --- Process scenes ---
    h, w = grid["shape"]
    emb_sum = np.zeros((h, w, 64), dtype=np.float64)
    emb_count = np.zeros((h, w), dtype=np.int32)
    files_dir = output_dir / f"{year}_files"
    used_scenes = []

    def _accumulate(emb, pixel_mask=None):
        valid = np.linalg.norm(emb, axis=-1) > 1e-6
        if pixel_mask is not None:
            valid &= pixel_mask
        emb_sum[valid] += emb[valid]
        emb_count[valid] += 1

    total_scenes = len(s2_items) + len(s1_items)
    scene_num = 0
    skipped = 0
    failed = 0

    for item in s2_items:
        scene_num += 1
        dt = item.datetime
        doy = dt.timetuple().tm_yday
        cc = item.properties.get("eo:cloud_cover", "?")
        mgrs = item.properties.get("s2:mgrs_tile", "???")
        log.info("[%d/%d] S2 %s %s (cloud=%.0f%%) ...", scene_num, total_scenes, mgrs, dt.date(), cc)

        try:
            t0 = time.time()
            s2_data = download_s2(item, grid)
            cov = check_coverage(s2_data)

            if cov < min_coverage:
                log.info("  Skipped: %.0f%% coverage < %.0f%% threshold", cov, min_coverage)
                skipped += 1
                continue

            # Per-pixel cloud/shadow mask from the SCL band. Falls back to
            # "keep all" if the SCL asset is missing (older S2 collections).
            cloud_mask = download_s2_cloud_mask(item, grid)
            if cloud_mask is not None:
                kept = float(cloud_mask.mean())
                log.info("  Downloaded (%.0f%% cov, %.0f%% pixels usable, %s)",
                         cov, 100 * kept, _elapsed(t0))
            else:
                log.info("  Downloaded (%.0f%% cov, no SCL mask, %s)", cov, _elapsed(t0))

            t0 = time.time()
            emb = model.predict(s2_l2a=s2_data, dem=dem, doy=doy,
                                tile_size=224, overlap=overlap)
            _accumulate(emb, pixel_mask=cloud_mask)
            log.info("  Predicted (%s)", _elapsed(t0))
        except Exception as e:  # noqa: BLE001
            log.warning("  FAILED: %s: %s — skipping scene", type(e).__name__, e)
            failed += 1
            continue

        # Save per-timestamp
        ts_dir = files_dir / f"{dt.date()}_s2"
        if save_per_timestamp_embedding or save_scenes:
            ts_dir.mkdir(parents=True, exist_ok=True)
        if save_per_timestamp_embedding:
            write_geotiff(emb.astype(np.float32), grid, ts_dir / "embedding.tif")
        if save_scenes:
            write_geotiff(s2_data, grid, ts_dir / "scene.tif", band_first=True)

        used_scenes.append({
            "sensor": "S2",
            "stac_collection": "sentinel-2-l2a",
            "stac_id": item.id,
            "date": str(dt.date()),
            "datetime": dt.isoformat(),
            "doy": doy,
            "mgrs": mgrs,
            "platform": item.properties.get("platform"),
            "cloud_cover": float(cc) if isinstance(cc, (int, float)) else None,
            "coverage": round(cov, 1),
        })

    for item in s1_items:
        scene_num += 1
        dt = item.datetime
        doy = dt.timetuple().tm_yday
        log.info("[%d/%d] S1 %s ...", scene_num, total_scenes, dt.date())

        try:
            t0 = time.time()
            s1_data = download_s1(item, grid)
            cov = check_coverage(s1_data)

            if cov < min_coverage:
                log.info("  Skipped: %.0f%% coverage < %.0f%% threshold", cov, min_coverage)
                skipped += 1
                continue

            log.info("  Downloaded (%.0f%% coverage, %s)", cov, _elapsed(t0))

            t0 = time.time()
            emb = model.predict(s1=s1_data, dem=dem, doy=doy,
                                tile_size=224, overlap=overlap)
            _accumulate(emb)
            log.info("  Predicted (%s)", _elapsed(t0))
        except Exception as e:  # noqa: BLE001
            log.warning("  FAILED: %s: %s — skipping scene", type(e).__name__, e)
            failed += 1
            continue

        ts_dir = files_dir / f"{dt.date()}_s1"
        if save_per_timestamp_embedding or save_scenes:
            ts_dir.mkdir(parents=True, exist_ok=True)
        if save_per_timestamp_embedding:
            write_geotiff(emb.astype(np.float32), grid, ts_dir / "embedding.tif")
        if save_scenes:
            write_geotiff(s1_data, grid, ts_dir / "scene.tif", band_first=True)

        used_scenes.append({
            "sensor": "S1",
            "stac_collection": "sentinel-1-rtc",
            "stac_id": item.id,
            "date": str(dt.date()),
            "datetime": dt.isoformat(),
            "doy": doy,
            "platform": item.properties.get("platform"),
            "orbit_state": item.properties.get("sat:orbit_state"),
            "polarizations": item.properties.get("sar:polarizations"),
            "instrument_mode": item.properties.get("sar:instrument_mode"),
            "coverage": round(cov, 1),
        })

    # --- Summary ---
    log.info("")
    log.info("=" * 50)
    log.info("  Scenes found:    %d", total_scenes)
    log.info("  Scenes used:     %d (>= %.0f%% coverage)", len(used_scenes), min_coverage)
    log.info("  Scenes skipped:  %d (below coverage threshold)", skipped)
    log.info("  Scenes failed:   %d (download/inference error)", failed)

    covered = emb_count > 0
    if not covered.any():
        log.error("No pixel coverage — no scenes passed the coverage filter.")
        log.error("Try lowering --min_coverage (currently %.0f%%).", min_coverage)
        return None, used_scenes

    n_covered = covered.sum()
    log.info("  Pixel coverage:  %d / %d (%.1f%%)", n_covered, h * w, 100 * n_covered / (h * w))
    log.info("  Obs per pixel:   min=%d, median=%.0f, max=%d",
             emb_count[covered].min(), np.median(emb_count[covered]), emb_count[covered].max())

    # --- Average + L2-normalise ---
    avg = np.zeros((h, w, 64), dtype=np.float32)
    avg[covered] = (emb_sum[covered] / emb_count[covered, np.newaxis]).astype(np.float32)
    norms = np.linalg.norm(avg, axis=-1, keepdims=True)
    avg = avg / np.clip(norms, 1e-8, None)
    avg[~covered] = 0

    # --- Write annual average ---
    out_path = output_dir / f"{year}.tif"
    write_geotiff(avg, grid, out_path)
    size_mb = out_path.stat().st_size / 1e6
    log.info("  Annual average:  %s (%.0f MB)", out_path, size_mb)

    # --- PCA previews (shared basis) ---
    log.info("  Generating PCA previews ...")
    pca_state = fit_pca(avg)
    write_pca_preview(avg, output_dir / f"{year}_preview_pca.png", pca_state=pca_state)

    if files_dir.exists():
        import rasterio
        for ts_dir in sorted(files_dir.iterdir()):
            emb_tif = ts_dir / "embedding.tif"
            if emb_tif.exists():
                with rasterio.open(emb_tif) as src:
                    ts_emb = src.read().transpose(1, 2, 0)
                write_pca_preview(ts_emb, ts_dir / "preview_pca.png", pca_state=pca_state)

    # --- Manifest ---
    # Try to extract model provenance (best-effort — attributes vary by variant)
    model_info: dict = {}
    for attr in ("repo_id", "_repo_id", "variant", "_variant", "model_type", "_model_type"):
        val = getattr(model, attr, None)
        if val is not None:
            model_info[attr.lstrip("_")] = str(val)
    try:
        import betaearth as _be
        model_info["betaearth_version"] = getattr(_be, "__version__", "unknown")
    except Exception:
        pass

    from datetime import timezone as _tz
    manifest = {
        "generated_at": datetime.now(_tz.utc).isoformat().replace("+00:00", "Z"),
        "model": model_info,
        "bbox_4326": list(bbox_4326),
        "crs": f"EPSG:{grid['epsg']}",
        "bounds": list(grid["bounds"]),
        "shape": list(grid["shape"]),
        "resolution_m": RESOLUTION,
        "year": year,
        "acquisition": {
            "stac_endpoint": PC_CATALOG,
            "collection_s2": "sentinel-2-l2a",
            "collection_s1": "sentinel-1-rtc",
            "max_cloud_cover_pct": max_cloud,
            "max_per_quarter_s2": max_per_quarter,
            "max_per_quarter_s1": 1,
            "min_coverage_pct": min_coverage,
        },
        "inference": {
            "tile_size": 224,
            "overlap": overlap,
        },
        "n_scenes_found": total_scenes,
        "n_scenes_used": len(used_scenes),
        "n_scenes_skipped": skipped,
        "n_scenes_failed": failed,
        "scenes": used_scenes,
    }
    manifest_path = output_dir / f"{year}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("  Manifest:        %s", manifest_path)
    log.info("=" * 50)
    return avg, used_scenes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate BetaEarth embeddings for any area of interest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Bavarian Forest National Park (annual mosaic only)
  betaearth-generate --osm_relation 1864214 --years 2023 --no_per_timestamp_embedding

  # Custom bounding box (west south east north)
  betaearth-generate --bbox 13.18 48.86 13.65 49.13 --years 2022 2023

  # Keep raw input scenes too
  betaearth-generate --bbox 13.18 48.86 13.65 49.13 --years 2023 --save_scenes
""",
    )
    # AOI
    aoi = parser.add_mutually_exclusive_group(required=True)
    aoi.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                     help="Bounding box in EPSG:4326")
    aoi.add_argument("--geojson", type=str, help="GeoJSON file with AOI polygon")
    aoi.add_argument("--osm_relation", type=int,
                     help="OSM relation ID (auto-pads bbox by ~1km)")

    # Time
    parser.add_argument("--years", nargs="+", type=int, required=True)

    # Scene selection
    parser.add_argument("--max_cloud", type=int, default=20,
                        help="Max cloud cover %% for S2 (default: 20)")
    parser.add_argument("--max_per_quarter", type=int, default=6,
                        help="Max scenes per quarter (default: 6)")
    parser.add_argument("--min_coverage", type=float, default=100.0,
                        help="Min AOI coverage %% to use a scene (default: 100)")

    # Model
    parser.add_argument("--model", default="asterisk-labs/betaearth-segformer-film-robust",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlap", type=int, default=112,
                        help="Tile overlap in pixels (default: 112)")

    # Output
    parser.add_argument("--output_dir", default="outputs",
                        help="Base output directory (default: outputs)")
    parser.add_argument("--name", default=None,
                        help="Subdirectory name (default: auto from AOI)")
    parser.add_argument("--save_scenes", action="store_true",
                        help="Save raw input scenes alongside embeddings")
    parser.add_argument("--no_per_timestamp_embedding", action="store_true",
                        help="Skip per-scene embedding GeoTIFFs (keep only annual average)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # --- Resolve AOI ---
    bbox_4326 = resolve_aoi(args)
    grid = compute_grid(bbox_4326)
    h, w = grid["shape"]
    log.info("AOI: (%.3f, %.3f) to (%.3f, %.3f)", *bbox_4326)
    log.info("Grid: %d x %d px (%.1f x %.1f km), EPSG:%d",
             w, h, w * RESOLUTION / 1000, h * RESOLUTION / 1000, grid["epsg"])

    # --- Output dir ---
    name = args.name
    if name is None:
        if args.osm_relation:
            name = f"osm_{args.osm_relation}"
        else:
            name = f"bbox_{bbox_4326[0]:.2f}_{bbox_4326[1]:.2f}_{bbox_4326[2]:.2f}_{bbox_4326[3]:.2f}"
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", output_dir)

    # --- Load model ---
    log.info("Loading model: %s ...", args.model)
    t0 = time.time()
    from betaearth import BetaEarth

    model = BetaEarth.from_pretrained(args.model, device=args.device)
    log.info("  %s (%s)", model, _elapsed(t0))

    # --- Generate per year ---
    for year in args.years:
        log.info("")
        log.info("=" * 50)
        log.info("  Year: %d", year)
        log.info("=" * 50)

        out_tif = output_dir / f"{year}.tif"
        if out_tif.exists():
            log.info("  Already exists: %s — skipping (delete to regenerate)", out_tif)
            continue

        generate(
            bbox_4326, year, model, grid, output_dir,
            max_cloud=args.max_cloud,
            max_per_quarter=args.max_per_quarter,
            min_coverage=args.min_coverage,
            overlap=args.overlap,
            save_scenes=args.save_scenes,
            save_per_timestamp_embedding=not args.no_per_timestamp_embedding,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
