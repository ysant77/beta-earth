"""BetaEarth Embedding Generator — Streamlit Demo.

Interactive map-based interface for generating dense 10m geospatial
embeddings. Deployable on HuggingFace Spaces.

Usage:
    cd demo && streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import folium
import folium.plugins
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BetaEarth",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOLUTION = 10.0
MAX_OUTPUT_MB = 3000
BYTES_PER_PIXEL = 64 * 4
COMPRESSION_RATIO = 1.0   # embeddings are near-incompressible (L2-normed float32)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @font-face {
        font-family: 'League Spartan';
        src: url('https://raw.githubusercontent.com/asterisk-labs/asterisk-labs.github.io/main/assets/fonts/league-spartan-v11-latin_latin-ext-regular.woff2') format('woff2');
        font-weight: 400;
        font-style: normal;
    }
    @font-face {
        font-family: 'League Spartan';
        src: url('https://raw.githubusercontent.com/asterisk-labs/asterisk-labs.github.io/main/assets/fonts/league-spartan-v11-latin_latin-ext-500.woff2') format('woff2');
        font-weight: 500;
        font-style: normal;
    }
    @font-face {
        font-family: 'League Spartan';
        src: url('https://raw.githubusercontent.com/asterisk-labs/asterisk-labs.github.io/main/assets/fonts/league-spartan-v11-latin_latin-ext-600.woff2') format('woff2');
        font-weight: 600;
        font-style: normal;
    }
    @font-face {
        font-family: 'League Spartan';
        src: url('https://raw.githubusercontent.com/asterisk-labs/asterisk-labs.github.io/main/assets/fonts/league-spartan-v11-latin_latin-ext-700.woff2') format('woff2');
        font-weight: 700;
        font-style: normal;
    }

    /* Base — force League Spartan everywhere */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
    .stRadio label, .stCheckbox label, .stSlider label,
    .stSelectSlider label, .stDateInput label, .stTextInput label,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
    [data-baseweb="select"], [data-baseweb="input"], [data-baseweb="radio"] label,
    [data-baseweb="toggle"] + div,
    button, input, select, textarea,
    .stCaption, .stAlert, .stToast,
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        font-family: 'League Spartan', sans-serif !important;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #ffffff !important;
    }
    .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #fafafa !important;
        border-right: 1px solid #e8e8e8 !important;
    }
    [data-testid="collapsedControl"],
    button[kind="headerNoPadding"] {
        display: none !important;
    }
    [data-testid="stSidebar"] h1 {
        font-family: 'League Spartan', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #888 !important;
        font-size: 0.8rem !important;
    }

    /* Metrics — yellow accent border */
    [data-testid="stMetric"] {
        background: #fffef5;
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #f7cc09;
        border-left: 4px solid #f7cc09;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #492ae8 !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #c4ffc2 !important;
    }

    /* Buttons — primary: blue, download: green */
    .stButton > button[kind="primary"] {
        background: #492ae8 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3a1fd0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(73,42,232,0.3) !important;
    }
    .stDownloadButton > button {
        background: #c4ffc2 !important;
        color: #1a1a1a !important;
        border: 2px solid #492ae8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        background: #a8f0a6 !important;
        transform: translateY(-1px) !important;
    }

    /* Progress bar — only target the fill element, not the text */
    .stProgress [role="progressbar"] > div:first-child {
        background: linear-gradient(90deg, #492ae8, #c4ffc2) !important;
        border-radius: 4px !important;
    }
    .stProgress {
        margin: 8px 0 !important;
    }
    .stProgress p {
        font-size: 0.85rem !important;
        color: #1a1a1a !important;
        margin-bottom: 4px !important;
    }

    /* Slider track fill — blue */
    .stSlider div[data-baseweb="slider"] div[role="progressbar"] {
        background: #492ae8 !important;
    }

    /* Toggle track — blue when on */
    [data-baseweb="toggle"] > div:first-child {
        background-color: #492ae8 !important;
    }

    /* Info boxes — soft purple */
    .stAlert [data-baseweb="notification"] {
        background: #f0eeff !important;
        border-left-color: #492ae8 !important;
    }

    /* Leaflet draw toolbar — larger icons */
    .leaflet-draw-toolbar a {
        width: 36px !important;
        height: 36px !important;
        line-height: 36px !important;
        background-size: 360px 30px !important;
    }
    .leaflet-draw-toolbar {
        margin-top: 6px !important;
    }

    /* Gallery images */
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Dividers */
    hr { border-color: #eee !important; }

    /* Badge / tag style for bbox coords */
    code {
        background: #f0eeff !important;
        color: #492ae8 !important;
        border-radius: 4px !important;
        font-size: 0.75rem !important;
    }

    /* Section headers below map */
    h3 { color: #492ae8 !important; font-weight: 700 !important; }

    /* Bottom panel — dark console look */
    [data-testid="stBottomBlockContainer"],
    .main > div:last-child {
        background: #1a1a2e !important;
        border-top: 3px solid #492ae8 !important;
        padding: 16px 24px !important;
    }

    /* Console-style text in bottom area */
    .console-panel {
        background: #1a1a2e;
        border-top: 3px solid #492ae8;
        border-radius: 0;
        padding: 20px 28px;
        margin: 0 -1rem;
        color: #e0e0e0;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
    }
    .console-panel h3 {
        color: #c4ffc2 !important;
        font-family: 'League Spartan', sans-serif !important;
        font-size: 1.1rem !important;
        margin-bottom: 12px !important;
    }
    .console-panel .stProgress [role="progressbar"] > div:first-child {
        background: linear-gradient(90deg, #492ae8, #c4ffc2) !important;
    }
    .console-panel p, .console-panel span {
        color: #e0e0e0 !important;
    }
    .console-panel code {
        background: #2a2a4a !important;
        color: #f7cc09 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------
def estimate_size(bbox, n_scenes=7):
    w, s, e, n = bbox
    width_km = (e - w) * 111 * abs(np.cos(np.radians((s + n) / 2)))
    height_km = (n - s) * 111
    n_pixels = int(width_km * 1000 / RESOLUTION) * int(height_km * 1000 / RESOLUTION)
    file_mb = n_pixels * BYTES_PER_PIXEL / COMPRESSION_RATIO / 1e6
    total_mb = file_mb * (1 + n_scenes)
    return round(width_km, 1), round(height_km, 1), round(total_mb, 1)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🥕 BetaEarth")
    st.caption("Unofficial AlphaEarth Emulator for Sentinel-2 and Sentinel-1 Embeddings")
    st.divider()

    time_mode = st.radio("Time range", ["Annual", "Custom dates"], horizontal=True)

    if time_mode == "Annual":
        year_range = st.slider("Years", 2017, 2025, (2023, 2023))
        years = list(range(year_range[0], year_range[1] + 1))
        custom_dates = None
    else:
        from datetime import date
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start", date(2023, 1, 1), min_value=date(2017, 1, 1))
        with col_d2:
            end_date = st.date_input("End", date(2023, 12, 31), max_value=date(2025, 12, 31))
        years = sorted(set(range(start_date.year, end_date.year + 1)))
        custom_dates = (str(start_date), str(end_date))

    min_coverage = st.slider("Min scene coverage (%)", 50, 100, 100, step=5)
    max_cloud = st.slider("Max cloud cover (%)", 5, 50, 20, step=5)
    save_per_timestamp = st.toggle("Save per-timestamp embeddings", value=True,
                                    help="Disable to only output the annual average (much smaller download)")

    st.divider()

    # Bbox display
    if "bbox" in st.session_state and st.session_state.bbox:
        bbox = st.session_state.bbox
        w_km, h_km, est_mb = estimate_size(bbox)
        n_years = len(years)
        if save_per_timestamp:
            total_mb = est_mb * n_years  # annual + per-timestamp
        else:
            per_year_mb = w_km * 1000 / RESOLUTION * h_km * 1000 / RESOLUTION * BYTES_PER_PIXEL / COMPRESSION_RATIO / 1e6
            total_mb = round(per_year_mb * n_years, 1)  # annual only
        st.metric("Area", f"{w_km} × {h_km} km")
        label = f"{total_mb} MB" + (f" ({n_years} yr)" if n_years > 1 else "")
        if total_mb <= MAX_OUTPUT_MB:
            st.metric("Est. output", label, delta="OK", delta_color="normal")
        else:
            st.metric("Est. output", label, delta=f">{MAX_OUTPUT_MB} MB!", delta_color="inverse")
        st.code(f"W={bbox[0]:.4f}\nS={bbox[1]:.4f}\nE={bbox[2]:.4f}\nN={bbox[3]:.4f}", language=None)
    else:
        st.info("Draw a rectangle on the map")

    st.divider()

    generate_btn = st.button(
        "🚀 Generate Embeddings",
        type="primary",
        use_container_width=True,
        disabled="bbox" not in st.session_state or not st.session_state.bbox,
    )

    if "results" in st.session_state and st.session_state.results:
        st.divider()
        st.caption(st.session_state.results["summary"])



# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
m = folium.Map(
    location=[48.5, 10.0],
    zoom_start=5,
    tiles=None,
    control_scale=True,
)

# Satellite (togglable)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
    overlay=False,
).add_to(m)

# Stamen Terrain (togglable)
folium.TileLayer(
    tiles="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png",
    attr="Stamen/Stadia",
    name="Terrain",
    overlay=False,
).add_to(m)

# Stamen Toner Lite — default (added last = shown first in Folium)
folium.TileLayer(
    tiles="https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png",
    attr="Stamen/Stadia",
    name="Toner Lite",
    overlay=False,
    show=True,
).add_to(m)

# Draw control for bbox
folium.plugins.Draw(
    draw_options={
        "polyline": False, "polygon": False, "circle": False,
        "circlemarker": False, "marker": False,
        "rectangle": {"shapeOptions": {"color": "#ff6600", "weight": 3, "fillOpacity": 0.1}},
    },
    edit_options={"edit": False},
).add_to(m)

# Auto-activate rectangle draw tool on map load
if "bbox" not in st.session_state or not st.session_state.bbox:
    auto_draw_js = folium.Element("""
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        setTimeout(function() {
            var btn = document.querySelector('.leaflet-draw-draw-rectangle');
            if (btn) btn.click();
        }, 500);
    });
    </script>
    """)
    m.get_root().html.add_child(auto_draw_js)

# Add PCA overlay if results exist
if "results" in st.session_state and st.session_state.results:
    res = st.session_state.results
    opacity = st.session_state.get("opacity", 0.7)
    bbox = res["bbox"]
    previews = res.get("previews", [])

    # Find the selected preview frame
    selected_label = st.session_state.get("preview_frame")
    preview_path = None
    if previews:
        preview_map = {label: path for path, label in previews}
        preview_path = preview_map.get(selected_label, previews[0][0])

    if preview_path and Path(preview_path).exists():
        import base64
        img_data = base64.b64encode(Path(preview_path).read_bytes()).decode()
        img_url = f"data:image/png;base64,{img_data}"
        folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            opacity=opacity,
            name="BetaEarth PCA",
        ).add_to(m)

    # Zoom to bbox
    m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

folium.LayerControl().add_to(m)

# Render map (full width)
map_data = st_folium(m, height=850, use_container_width=True, returned_objects=["all_drawings"])

# Extract bbox from drawn rectangle
if map_data and map_data.get("all_drawings"):
    drawings = map_data["all_drawings"]
    if drawings:
        last = drawings[-1]
        if last["geometry"]["type"] == "Polygon":
            coords = last["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            new_bbox = (min(lons), min(lats), max(lons), max(lats))
            if st.session_state.get("bbox") != new_bbox:
                st.session_state.bbox = new_bbox
                st.rerun()  # rerun so sidebar picks up the new bbox immediately


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
if generate_btn and "bbox" in st.session_state and st.session_state.bbox:
    bbox = st.session_state.bbox
    w_km, h_km, est_mb = estimate_size(bbox)

    n_years = len(years)
    if save_per_timestamp:
        total_mb = est_mb * n_years
    else:
        w_km, h_km, _ = estimate_size(bbox, n_scenes=0)
        per_year_mb = w_km * 1000 / RESOLUTION * h_km * 1000 / RESOLUTION * BYTES_PER_PIXEL / COMPRESSION_RATIO / 1e6
        total_mb = round(per_year_mb * n_years, 1)
    if total_mb > MAX_OUTPUT_MB:
        st.error(f"Estimated output ({total_mb:.0f} MB) exceeds {MAX_OUTPUT_MB} MB limit. Select a smaller region or fewer years.")
    else:
        progress = st.progress(0, text="Loading model...")

        # Lazy imports
        from betaearth import BetaEarth
        import torch
        from examples.generate import (
            compute_grid, download_dem, download_s2, download_s1,
            _search_stac, _seasonal_select, check_coverage,
            write_geotiff, fit_pca, write_pca_preview,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BetaEarth.from_pretrained(device=device)
        progress.progress(5, text=f"Model loaded on {device}")

        grid = compute_grid(bbox)
        h, w = grid["shape"]

        run_id = uuid.uuid4().hex[:8]
        output_dir = Path(tempfile.mkdtemp()) / f"betaearth_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # DEM (shared across years)
        progress.progress(8, text="Downloading DEM...")
        dem = download_dem(grid)

        all_previews = []
        all_summaries = []
        last_annual_preview = None

        for yi, year in enumerate(years):
            # Progress range for this year: spread evenly across [10, 90]
            yr_lo = 10 + int(80 * yi / n_years)
            yr_hi = 10 + int(80 * (yi + 1) / n_years)
            yr_label = f"[{year}] " if n_years > 1 else ""

            files_dir = output_dir / f"{year}_files"
            files_dir.mkdir()

            # Search
            progress.progress(yr_lo, text=f"{yr_label}Searching Planetary Computer...")
            s2_items = _search_stac(bbox, year, "sentinel-2-l2a", max_cloud=max_cloud)
            s2_items = _seasonal_select(s2_items, max_per_quarter=6)
            s1_items = _search_stac(bbox, year, "sentinel-1-rtc")
            s1_items = _seasonal_select(s1_items, max_per_quarter=1, use_cloud=False)
            total_found = len(s2_items) + len(s1_items)

            # Process scenes
            emb_sum = np.zeros((h, w, 64), dtype=np.float64)
            emb_count = np.zeros((h, w), dtype=np.int32)
            used_scenes = []
            all_items = [(it, "S2") for it in s2_items] + [(it, "S1") for it in s1_items]
            n_total = len(all_items)
            skipped = 0

            for i, (item, sensor) in enumerate(all_items):
                pct = yr_lo + int((yr_hi - yr_lo - 10) * i / max(n_total, 1))
                dt = item.datetime
                doy = dt.timetuple().tm_yday

                if sensor == "S2":
                    mgrs = item.properties.get("s2:mgrs_tile", "???")
                    cc = item.properties.get("eo:cloud_cover", 0)
                    progress.progress(pct, text=f"{yr_label}[{i+1}/{n_total}] S2 {mgrs} {dt.date()} (cloud={cc:.0f}%)")
                    data = download_s2(item, grid)
                else:
                    progress.progress(pct, text=f"{yr_label}[{i+1}/{n_total}] S1 {dt.date()}")
                    data = download_s1(item, grid)

                cov = check_coverage(data)
                if cov < min_coverage:
                    skipped += 1
                    continue

                progress.progress(pct + 1, text=f"{yr_label}[{i+1}/{n_total}] Predicting {sensor} {dt.date()}...")
                if sensor == "S2":
                    emb = model.predict(s2_l2a=data, dem=dem, doy=doy, tile_size=224, overlap=112)
                    ts_label = f"{dt.date()}_s2"
                else:
                    emb = model.predict(s1=data, dem=dem, doy=doy, tile_size=224, overlap=112)
                    ts_label = f"{dt.date()}_s1"

                valid = np.linalg.norm(emb, axis=-1) > 1e-6
                emb_sum[valid] += emb[valid]
                emb_count[valid] += 1

                ts_dir = files_dir / ts_label
                ts_dir.mkdir(parents=True, exist_ok=True)
                if save_per_timestamp:
                    write_geotiff(emb.astype(np.float32), grid, ts_dir / "embedding.tif")
                used_scenes.append({"sensor": sensor, "date": str(dt.date()), "doy": doy, "coverage": round(cov, 1)})

            if not used_scenes:
                all_summaries.append(f"**{year}:** No scenes passed {min_coverage}% filter")
                continue

            # Average
            progress.progress(yr_hi - 5, text=f"{yr_label}Averaging...")
            covered = emb_count > 0
            avg = np.zeros((h, w, 64), dtype=np.float32)
            avg[covered] = (emb_sum[covered] / emb_count[covered, np.newaxis]).astype(np.float32)
            norms = np.linalg.norm(avg, axis=-1, keepdims=True)
            avg = avg / np.clip(norms, 1e-8, None)
            avg[~covered] = 0
            write_geotiff(avg, grid, output_dir / f"{year}.tif")

            # PCA previews
            progress.progress(yr_hi - 2, text=f"{yr_label}PCA previews...")
            pca_state = fit_pca(avg)
            annual_preview = output_dir / f"{year}_preview_pca.png"
            write_pca_preview(avg, annual_preview, pca_state=pca_state)
            last_annual_preview = str(annual_preview)

            import rasterio
            for ts_dir in sorted(files_dir.iterdir()):
                emb_tif = ts_dir / "embedding.tif"
                if emb_tif.exists():
                    with rasterio.open(emb_tif) as src:
                        ts_emb = src.read().transpose(1, 2, 0)
                    write_pca_preview(ts_emb, ts_dir / "preview_pca.png", pca_state=pca_state)

            # Manifest
            manifest = {
                "bbox_4326": list(bbox), "year": year,
                "min_coverage": min_coverage,
                "n_scenes_found": total_found,
                "n_scenes_used": len(used_scenes),
                "scenes": used_scenes,
            }
            with open(output_dir / f"{year}_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            all_previews.append((str(annual_preview), f"{year} average"))
            for ts_dir in sorted(files_dir.iterdir()):
                png = ts_dir / "preview_pca.png"
                if png.exists():
                    all_previews.append((str(png), f"{year}/{ts_dir.name}"))

            all_summaries.append(
                f"**{year}:** {len(used_scenes)} scenes "
                f"({total_found} found, {skipped} skipped)"
            )

        # ZIP everything
        progress.progress(92, text="Creating download archive...")
        zip_path = output_dir.parent / f"betaearth_{run_id}"
        shutil.make_archive(str(zip_path), "zip", str(output_dir))
        zip_file = str(zip_path) + ".zip"
        with open(zip_file, "rb") as f:
            zip_data = f.read()

        st.session_state.results = {
            "bbox": list(bbox),
            "zip_data": zip_data,
            "zip_name": f"betaearth_{run_id}.zip",
            "annual_preview": last_annual_preview,
            "summary": "\n\n".join(all_summaries),
            "previews": all_previews,
        }

        progress.progress(100, text="Done!")
        st.rerun()


# ---------------------------------------------------------------------------
# Overlay controls + preview gallery (below map, only after generation)
# ---------------------------------------------------------------------------
if "results" in st.session_state and st.session_state.results:
    res = st.session_state.results
    previews = res.get("previews", [])
    if previews:
        labels = [label for _, label in previews]
        col_slider, col_opacity, col_dl = st.columns([3, 1, 1])
        with col_slider:
            st.select_slider("Preview frame", options=labels, value=labels[0], key="preview_frame")
        with col_opacity:
            st.slider("Opacity", 0.0, 1.0, 0.7, step=0.05, key="opacity")
        with col_dl:
            st.write("")  # vertical spacing to align with sliders
            st.download_button(
                "📦 Download ZIP",
                data=res["zip_data"],
                file_name=res["zip_name"],
                mime="application/zip",
                use_container_width=True,
            )

        st.subheader("PCA-RGB Previews")
        cols = st.columns(min(len(previews), 5))
        for i, (path, label) in enumerate(previews):
            with cols[i % len(cols)]:
                if Path(path).exists():
                    st.image(path, caption=label, use_container_width=True)
