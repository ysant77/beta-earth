<img width="1897" height="800" alt="beta-final" src="https://github.com/user-attachments/assets/91b2a46c-a142-4eed-99ce-a1b692178146" />


[![Preprint PDF](https://img.shields.io/badge/Preprint-PDF-b31b1b.svg)](docs/beta_earth_preprint.pdf)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset](https://img.shields.io/badge/🤗-Major_TOM-yellow)](https://huggingface.co/Major-TOM)
[![Browser Demo](https://img.shields.io/badge/🤗-Browser_Demo-ff9d00)](https://huggingface.co/spaces/asterisk-labs/betaearth)
[![ISPRS 2026](https://img.shields.io/badge/ISPRS-2026-blue)](https://www.isprs.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asterisk-labs/beta-earth/blob/main/examples/demo.ipynb)
[![PyPI](https://img.shields.io/pypi/v/betaearth)](https://pypi.org/project/betaearth/)

**Embedding Sentinel-2 and Sentinel-1 with a Little Help of AlphaEarth**

> 📄 **Official paper coming soon.** The write-up (architecture, evaluation on 6,250 test tiles, modality attribution, multi-temporal aggregation) will be published on **EarthArXiv** shortly. Working draft available as a local PDF: [`docs/beta_earth_preprint.pdf`](docs/beta_earth_preprint.pdf).

---

## What is BetaEarth?
<img width="2978" height="1000" alt="2023_preview_pca" src="https://github.com/user-attachments/assets/de8baf2d-f24e-4dd7-8db4-1d5fa562d7d8" />

BetaEarth produces dense 10 m geospatial embedding fields from Sentinel-2 and Sentinel-1 imagery. It is trained to approximate the outputs of [AlphaEarth Foundations](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) (AEF) — the embedding product released by Google and Google DeepMind — using only AEF's public precomputed embeddings as supervision.

BetaEarth has no access to AEF's weights or architecture. It is an independent model, not a variant or extension of AEF. Emulation quality is below AEF's, but BetaEarth runs locally on any Sentinel scene and its full pipeline is open.

<img width="1200" height="630" alt="beta-earth" src="https://github.com/user-attachments/assets/848e21a3-7af9-4614-898a-8a11a2015eff" />

### When to use BetaEarth

- **Offline generation.** AEF is distributed as annual global rasters generated inside Google Earth Engine. BetaEarth runs on any S2/S1 scene locally — useful for custom temporal windows or deployments without Earth Engine access.
- **Open pipeline.** Training data, weights, and inference code are all open, so BetaEarth can serve as an approximate reference for studying how multimodal Earth-observation embeddings behave under missing modalities, temporal averaging, or compression.

---

## Quickstart

```bash
pip install betaearth
```

```python
from betaearth import BetaEarth

model = BetaEarth.from_pretrained()  # default: curriculum flagship (HF repo: betaearth-segformer-film-robust)

# Any modality can be omitted — the curriculum model handles missing inputs.
# predict() tiles internally (224 px tile, 112 px overlap, trapezoidal blend),
# so any (H, W) works — including full 1068x1068 Major TOM tiles or larger.
embedding = model.predict(
    s2_l2a=s2_l2a,   # (9, H, W) uint16 DN; bands [B02,B03,B04,B08,B05,B06,B07,B11,B12]
    s2_l1c=s2_l1c,   # (9, H, W) uint16 DN; same band order as L2A
    s1=s1,           # (2, H, W) float32 linear power (NOT dB); bands [VV, VH]
    dem=dem,         # (1, H, W) float32 elevation in metres (raw COP-DEM)
    doy=182,         # day-of-year of the S2 acquisition (1-366)
)
# embedding: (H, W, 64) float32, L2-normalised per pixel (unit vectors on S^63)
```

### Input formats

All spatial arrays share the same `(H, W)` and are pixel-aligned. BetaEarth normalises internally — **pass the raw source values** described below, no custom scaling.

| Input | Shape | Dtype | Units / range | Band order | Typical source |
|---|---|---|---|---|---|
| `s2_l1c` | `(9, H, W)` | uint16 | Digital numbers, 0–10 000+ (top-of-atmosphere reflectance × 10 000). Divided by 10 000 internally. | `[B02, B03, B04, B08, B05, B06, B07, B11, B12]` | Copernicus Data Space Ecosystem, Sentinel Hub, AWS Open Data |
| `s2_l2a` | `(9, H, W)` | uint16 | Digital numbers, 0–10 000+ (atmospherically-corrected surface reflectance × 10 000). Divided by 10 000 internally. | same as L1C | Planetary Computer, Sentinel Hub, AWS Earth Search |
| `s1` | `(2, H, W)` | float32 | **Linear power** (typical range ~0–200, not 0–1). Converted to dB and rescaled internally. | `[VV, VH]` | Planetary Computer `sentinel-1-rtc`, ASF Radiometric Terrain Corrected |
| `dem` | `(1, H, W)` | float32 | Raw elevation in metres (COP-DEM GLO-30 range ~−500 to 9000). Min-max rescaled internally. | – | Copernicus DEM GLO-30 (Planetary Computer `cop-dem-glo-30`) |
| `doy` | scalar int | 1–366 | Day-of-year of the S2 acquisition (not epoch, not ISO) | – | – |

Output is `(H, W, 64) float32`, L2-normalised per pixel. `H` and `W` can be anything ≥ 224; `predict()` tiles the input with a 224×224 window internally and stitches.

### Input gotchas

- **S2 band order matters.** The 10 m bands come first, then 20 m: `[B02, B03, B04, B08, B05, B06, B07, B11, B12]`. Any other order silently produces garbage embeddings. If you fetch from a STAC source that returns bands in their native order (`B01, B02, …`), you must reorder before passing in.
- **L1C and L2A are NOT interchangeable.** They are handled by separate encoders and represent distinct processing levels (top-of-atmosphere vs surface reflectance). The default **curriculum (flagship)** model handles any subset (single L1C, single L2A, both, or neither) gracefully. The **peak-quality** variants (`betaearth-segformer-film` = reinit, `betaearth-segformer-film-hilr`, `betaearth-segformer-film-scratch`) were trained with **L1C + L2A jointly** and drop ~32 % cos sim if only one processing level is provided.
- **Raw DN values, not reflectance.** S2 normalisation happens inside the model — pass the uint16 DN as-is.
- **S1 must be linear power, not dB.** Planetary Computer's `sentinel-1-rtc` collection returns linear power by default. If you have GRD-dB data (e.g. from SNAP), convert first: `linear = 10 ** (db / 10)`. Typical linear-power magnitudes are ~0.01–200; `predict()` handles the dB conversion and clipping internally.
- **DEM in metres, not pre-normalised.** Pass the raw elevation array (COP-DEM GLO-30 output). `predict()` applies per-input min-max rescaling internally. If you already have DEM rescaled to `[0, 1]`, pass `normalise=False` to `predict()`.
- **Shape convention.** All spatial inputs are channel-first `(C, H, W)` — consistent with torch conventions but opposite of common remote-sensing `(H, W, C)` rasters.
- **Tiling is automatic.** `predict()` uses a 224 px tile with 112 px overlap (trapezoidal blending) by default — matches the paper's eval pipeline and gives seam-free PCA-RGB previews on low-variance scenes (arid, water, snow). Override with `tile_size=...` / `overlap=...` if you want a different stitch; `overlap=32` is ~3× faster but can show visible seams on uniform surfaces. Anything below 224 px total will fail.

**Try in 30 seconds on Colab** — pick the notebook that matches your use case:

| Notebook | When to use | Inputs | Runtime |
|---|---|---|---|
| ⚡ [`demo.ipynb`](examples/demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asterisk-labs/beta-earth/blob/main/examples/demo.ipynb) | **Fast mono-temporal quickstart.** Understand the model in one minute. | 1 Major TOM tile (single parquet row, no STAC) | ~30 s on T4 |
| 🌍 [`generate_demo.ipynb`](examples/generate_demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asterisk-labs/beta-earth/blob/main/examples/generate_demo.ipynb) | **Flexible multi-temporal** — any bounding box, annual aggregated embedding. Same pipeline as the hosted app. | S2 + S1 + DEM from Planetary Computer (multi-scene) | few minutes on T4 |

Or skip the notebooks: [`examples/predict.py`](examples/predict.py) is the minimal local script.

---

## Generate embeddings for any area


Four entry points, from zero-install to fully scripted.

### 1. Hosted app (no install)

Pick a bounding box on a map, click run. Free tier is CPU-only and caps total output at 3 GB.

<p align="center">
  <a href="https://huggingface.co/spaces/asterisk-labs/betaearth">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg" alt="Open in HF Spaces" height="56"/>
  </a>
</p>

<img width="2399" height="1240" alt="BetaEarth App" src="https://github.com/user-attachments/assets/167f6a0f-3216-4e43-96aa-bdf1490a68b4" />

### 2. Colab notebooks

Two notebooks depending on how much acquisition plumbing you want:

- ⚡ [`examples/demo.ipynb`](examples/demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asterisk-labs/beta-earth/blob/main/examples/demo.ipynb) — **fast mono-temporal**. One Major TOM tile, one `predict()`, one PCA-RGB. No STAC, no credentials. Good for understanding the model.
- 🌍 [`examples/generate_demo.ipynb`](examples/generate_demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asterisk-labs/beta-earth/blob/main/examples/generate_demo.ipynb) — **flexible multi-temporal**. Pick any bbox on an interactive map; the notebook downloads Sentinel-2 L2A + Sentinel-1 RTC + COP-DEM from Planetary Computer, runs per-timestamp inference, averages into an annual 64-band GeoTIFF. Uses Colab's free T4 GPU. This is the same pipeline as the hosted Streamlit app.

### 3. Command-line generation (the main path for real work)

`betaearth-generate` ships with the package and drives the same pipeline: download Sentinel-2 L2A + Sentinel-1 RTC + COP-DEM from [Planetary Computer](https://planetarycomputer.microsoft.com), run tiled inference, write an annual 64-band COG plus a full provenance manifest per year.

```bash
pip install 'betaearth[generate]'

# By bounding box (W S E N), one or more years
betaearth-generate --bbox 13.1 48.7 13.8 49.2 --years 2020 2021 2022 2023 2024 2025 \
    --output_dir outputs/bavarian_forest

# By OSM relation id (resolved to its bbox)
betaearth-generate --osm_relation 1864214 --years 2024 --output_dir outputs/bav
```

No API keys needed — Planetary Computer is publicly accessible. A CUDA GPU is used automatically if available; CPU works but is slower. Each run produces, per year:

| File | Description |
|---|---|
| `{year}.tif` | 64-band annual average embedding (L2-normalised per pixel), COG |
| `{year}_preview_pca.png` | 3-band PCA-RGB quick-look of the annual mosaic |
| `{year}_manifest.json` | Provenance: model repo + version, CRS/bounds/shape, acquisition params, full STAC id list of every scene used (cloud cover, coverage, S1 orbit/polarisation, ...) |
| `{year}_files/{date}_{sensor}/` | Optional per-scene outputs, only with `--save_per_timestamp_embedding` / `--save_scenes` |

The manifest is deliberately verbose so any downstream user of the embedding can verify exactly which Sentinel products fed into it. Import `betaearth.generate` for the Python API that backs the CLI; a minimal scripted example is in [`examples/predict.py`](examples/predict.py).

### 4. Streamlit app (local)

The same app as the hosted Space, run on your own compute:

```bash
git clone https://github.com/asterisk-labs/beta-earth
cd beta-earth
pip install 'betaearth[demo]'
streamlit run demo/app.py
```

Then open http://localhost:8501 in your browser. Raise the 3 GB cap via env var:

```bash
BETAEARTH_MAX_OUTPUT_MB=50000 streamlit run demo/app.py   # 50 GB ceiling
```

## Models

We release **8 model variants** spanning different trade-offs between quality, parameter efficiency, and input requirements.

### Main results (full 6,250-tile test set)

> **Preliminary results from the first preprint version.** Numbers match the working draft ([`docs/beta_earth_preprint.pdf`](docs/beta_earth_preprint.pdf), Table II) — full test set; own-probe LULC. Subject to revision once the paper goes live on EarthArXiv and in subsequent versions as evaluation is expanded.

| Model | Test Cos Sim | Std | LULC Acc | Model Size | Inputs |
|---|:---:|:---:|:---:|---:|---|
| **SF curriculum (flagship)** | **0.873** | 0.109 | 0.833 | 104.8M | **Any subset** of S2/S1/DEM + DOY |
| SF frozen+FiLM (reinit) | **0.883** | 0.106 | 0.836 | 104.8M | S2 L1C+L2A, S1, DEM, DOY |
| SF frozen+FiLM (hilr) | 0.883 | 0.107 | 0.838 | 104.8M | S2 L1C+L2A, S1, DEM, DOY |
| SF from-scratch+FiLM | 0.883 | 0.105 | 0.835 | 104.8M | S2 L1C+L2A, S1, DEM, DOY |
| SF no FiLM (baseline) | 0.875 | 0.110 | 0.838 | 104.8M | S2 L1C+L2A, S1, DEM |
| DINOv3 ViT-L/16 | 0.873 | 0.109 | **0.840** | 304M | 6 primitives + DOY |
| DINOv3 ViT-S/16 | 0.862 | 0.112 | 0.836 | 24M | 6 primitives + DOY |
| SF RGB-only+FiLM | 0.834 | 0.128 | 0.823 | 26.3M | S2 RGB, DOY |
| *Real AlphaEarth (reference)* | *---* | *---* | *0.856* | --- | --- |

### Single-modality performance (curriculum flagship, test set)

> **Preliminary — working draft.** Values match the preprint Table III (curriculum on the full 6,250-tile test set). See [`docs/beta_earth_preprint.pdf`](docs/beta_earth_preprint.pdf).

The **curriculum** model is the only variant that remains functional under severely reduced inputs:

| Input subset | Cosine sim |
|---|:---:|
| All modalities | 0.872 |
| No DEM (S2+S1 only) | 0.854 |
| No S1 (S2+DEM only) | 0.848 |
| S2 only | 0.817 |
| No time (DOY=0) | 0.773 |
| S1 only | 0.710 |
| DEM only | 0.541 |

For users with access to only one S2 processing level, separate validation-set measurements give **L1C-only 0.806** and **L2A-only 0.755** (the paper's test-set ablation groups both L1C and L2A together under "S2 only").

### Which model should I use?

| Use case | Recommended model | Why |
|---|---|---|
| **General use (default)** | SF curriculum (flagship) | Works with any input subset; only variant that stays usable on single-modality inputs (S1-only 0.710, DEM-only 0.541) |
| **Maximum quality** | SF frozen+FiLM (reinit) | Highest test cos sim (0.883) — requires all 4 modalities |
| **No timestamp needed** | SF no FiLM (baseline) | Does not consume day-of-year input; reaches 0.875 |
| **Lightweight / edge** | DINOv3 ViT-S/16 | 24M params, 0.862 test cos sim |
| **Minimal data requirements** | SF RGB-only+FiLM | Only needs 3-band S2 RGB + DOY |
| **Best downstream LULC** | DINOv3 ViT-L/16 | 0.840 own-probe LULC (closest to AEF's 0.856 ceiling) |
| **Research / ablation** | SF frozen+FiLM (hilr), SF from-scratch+FiLM | Alternative training strategies for comparison against the reinit variant |

### Architecture overview

**DINOv3 models** use a single shared frozen DINOv3 backbone applied to 3-band spectral primitives:

| Primitive | Bands | Captures |
|---|---|---|
| True-colour RGB | B04/B03/B02 | Visual texture, built environment |
| False-colour IR | B08/B04/B03 | Vegetation health (NIR) |
| SWIR composite | B12/B11/B04 | Moisture, bare soil, burn scars |
| Red-edge | B07/B06/B05 | Canopy structure, chlorophyll |
| SAR | VV/VH/ratio | Structure, moisture (from S1) |
| Topography | Elevation/Slope/Aspect | Terrain (from COP-DEM) |

Primitives are fused via permutation-invariant cross-attention (SetFusion).

**SegFormer models** use 4 separate MiT-B2 encoders processing each modality's raw bands natively (9ch S2-L1C, 9ch S2-L2A, 2ch S1, 1ch DEM), with channel concatenation fusion.

All models use **FiLM temporal conditioning** (day-of-year modulation) except the no-FiLM baseline.

### Key findings

- **Temporal conditioning as spectral compensation:** FiLM importance scales inversely with spectral access — RGB-only (22pp) > DINOv3 (18pp) > SegFormer scratch (14pp) > frozen SegFormer (5pp).
- **Multi-temporal averaging** of 4+ observations improves emulation by up to +13pp over single timestamps, with the benefit biome-dependent (gap-fill wins in boreal regions; S2-only wins in arid/temperate).
- Predicted embeddings retain **97% of downstream LULC classification accuracy** (own-probe linear probe on IO-LULC) across all full-spectrum variants.

---

## Model Properties

| Property | Value |
|---|---|
| Output | Dense embedding field — `(H, W, 64)` per tile at 10m resolution |
| Output normalisation | L2-normalised per pixel (unit vectors on S^63) |
| Quantisation | Original AEF: int8 on S^63; BetaEarth outputs float32 |
| Tile size | 10.68 x 10.68 km (1068 x 1068 px), Major TOM grid |
| Training data | 62,489 Major TOM grid cells (49,991 train / 6,248 val / 6,250 test) |
| Loss | Cosine similarity + 0.1 * MSE, masked to valid pixels |

---

## Multi-temporal averaging

Build an annual mosaic by predicting each scene separately and averaging the L2-normalised outputs — saturates at ~4 observations per pixel:

```python
import numpy as np

preds = []
for s2, s1, doy in zip(s2_timeseries, s1_timeseries, doys):
    pred = model.predict(s2_l2a=s2, s1=s1, dem=dem, doy=doy)
    preds.append(pred)

annual = np.mean(preds, axis=0)
annual /= np.linalg.norm(annual, axis=-1, keepdims=True)
```

(`betaearth-generate` and the Streamlit demo wrap this pattern with cloud masking, seasonal balancing, and a provenance manifest.)

---

## Data Access

All training data is from the [Major TOM](https://huggingface.co/Major-TOM) community project and is freely available on HuggingFace:

| Dataset | Description |
|---|---|
| [Major-TOM/Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A) | Sentinel-2 L2A imagery |
| [Major-TOM/Core-S2L1C](https://huggingface.co/datasets/Major-TOM/Core-S2L1C) | Sentinel-2 L1C imagery |
| [Major-TOM/Core-S1RTC](https://huggingface.co/datasets/Major-TOM/Core-S1RTC) | Sentinel-1 RTC imagery |
| [Major-TOM/Core-AlphaEarth-Embeddings](https://huggingface.co/datasets/Major-TOM/Core-AlphaEarth-Embeddings) | AEF target embeddings |

### Data normalisation

All input data should be stored as raw values. Normalisation happens inside the model:
- **S2 L1C/L2A:** uint16 DN (0-10000+), divided by 10000 internally
- **S1 RTC:** linear power (float32, ~0-200), log-transformed internally
- **COP-DEM:** raw elevation in metres (float32, COP-DEM GLO-30 range ~-500 to 9000), min-max rescaled internally (pass `normalise=False` to `predict()` if your DEM is already in `[0, 1]`)

**Important:** S2 bands must be ordered `[B02, B03, B04, B08, B05, B06, B07, B11, B12]` (10 m bands first, then 20 m) — the order BetaEarth was trained with.

---

## Citation

```bibtex
@inproceedings{czerkawski2026betaearth,
  title     = {BetaEarth: Emulating Closed-Source Earth Observation Models Through Their Public Embeddings},
  author    = {Czerkawski, Mikolaj},
  year      = {2026}
}
```

If using BetaEarth embeddings in research, also cite AlphaEarth Foundations ([arXiv:2507.22291](https://arxiv.org/abs/2507.22291)).

---

## License and Attribution

BetaEarth model weights are released under **CC-BY 4.0**, matching the license of the AlphaEarth Foundations embedding archive used for training supervision.

**Attribution for AEF training data:**
> "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."

Training imagery is sourced from Major TOM (Apache 2.0) and Copernicus Sentinel (free and open access).
