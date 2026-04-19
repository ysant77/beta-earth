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

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
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
HF_DATASET_REPO = "asterisk-labs/betaearth-requests"  # Private dataset for request logging


# ---------------------------------------------------------------------------
# Request logging to HuggingFace Dataset
# ---------------------------------------------------------------------------
def _log_request_async(
    timestamp: str,
    bbox: tuple[float, float, float, float],
    area_km2: float,
    years: list[int],
    time_mode: str,
    save_per_timestamp: bool,
    save_per_timestamp_input: bool,
) -> None:
    """Fire-and-forget logging of request metadata to HuggingFace Dataset."""
    try:
        from huggingface_hub import get_write_access_token
        from pathlib import Path
        import json
        import tempfile

        # Get HF token from Streamlit secrets (OAuth provider)
        # If running on HF Spaces, HUGGINGFACE_TOKEN or HF_TOKEN env variable should be available
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return  # Silent fail if no token available

        # Construct request record
        record = {
            "timestamp": timestamp,
            "bbox_w": bbox[0],
            "bbox_s": bbox[1],
            "bbox_e": bbox[2],
            "bbox_n": bbox[3],
            "area_km2": float(area_km2),
            "years": years,
            "time_mode": time_mode,
            "save_per_timestamp": bool(save_per_timestamp),
            "save_per_timestamp_input": bool(save_per_timestamp_input),
        }

        # Write as JSON to a temporary file, append to dataset via parquet
        import pandas as pd
        from huggingface_hub import CommitOperationAdd, HfApi, upload_file

        # Create a single-row parquet file
        df = pd.DataFrame([record])
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            tmp_path = tmp.name

        # Upload as a new commit to the dataset
        # Using CommitOperationAdd to append a timestamped file
        api = HfApi(token=hf_token)

        # Create a unique file name based on timestamp to avoid collisions
        timestamp_clean = timestamp.replace(":", "-").replace(".", "-")
        file_name = f"requests/{timestamp_clean}.parquet"

        # Upload the file
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=file_name,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Log request {timestamp}",
        )

        # Clean up temp file
        Path(tmp_path).unlink()
    except Exception as e:
        # Silent fail — don't interrupt user experience
        pass


def log_request(
    bbox: tuple[float, float, float, float],
    area_km2: float,
    years: list[int],
    time_mode: str,
    save_per_timestamp: bool,
    save_per_timestamp_input: bool,
) -> None:
    """Log request metadata asynchronously (fire-and-forget)."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    thread = threading.Thread(
        target=_log_request_async,
        args=(timestamp, bbox, area_km2, years, time_mode, save_per_timestamp, save_per_timestamp_input),
        daemon=True,
    )
    thread.start()



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
MIN_SIDE_KM = 3.2   # ~320 px at 10 m — safe multiple of 32 and > tile_size 224


def estimate_size(
    bbox,
    n_scenes=28,
    n_years=1,
    save_per_timestamp=True,
    save_per_timestamp_input=False,
):
    """Return (width_km, height_km, total_mb). Factors in per-timestamp toggles."""
    w, s, e, n = bbox
    width_km = (e - w) * 111 * abs(np.cos(np.radians((s + n) / 2)))
    height_km = (n - s) * 111
    n_pixels = int(width_km * 1000 / RESOLUTION) * int(height_km * 1000 / RESOLUTION)
    emb_bytes = n_pixels * BYTES_PER_PIXEL               # 64 bands × f32
    s2_bytes = n_pixels * 9 * 4                          # 9 bands × f32
    s1_bytes = n_pixels * 2 * 4                          # 2 bands × f32
    dem_bytes = n_pixels * 4                             # 1 band × f32

    # Rough split: ~24 S2 + ~4 S1 per year at max quota
    s2_per_year = 24 * (n_scenes / 28)
    s1_per_year = 4 * (n_scenes / 28)

    total = emb_bytes * n_years                          # 1 annual .tif per year
    if save_per_timestamp:
        total += emb_bytes * (s2_per_year + s1_per_year) * n_years
    if save_per_timestamp_input:
        total += s2_bytes * s2_per_year * n_years
        total += s1_bytes * s1_per_year * n_years
        total += dem_bytes                                # one-time
    return round(width_km, 1), round(height_km, 1), round(total / 1e6, 1)


def expand_to_min(bbox, min_km=MIN_SIDE_KM):
    """Expand bbox symmetrically so each side >= min_km. Returns (new_bbox, was_padded)."""
    w, s, e, n = bbox
    lat_mid = (s + n) / 2
    width_km = (e - w) * 111 * abs(np.cos(np.radians(lat_mid)))
    height_km = (n - s) * 111
    pad_x = pad_y = 0
    if width_km < min_km:
        pad_x = (min_km - width_km) / (2 * 111 * abs(np.cos(np.radians(lat_mid))))
    if height_km < min_km:
        pad_y = (min_km - height_km) / (2 * 111)
    was_padded = (pad_x > 0) or (pad_y > 0)
    return (w - pad_x, s - pad_y, e + pad_x, n + pad_y), was_padded


def crop_to_user(arr, pad_grid, user_grid, channel_axis=-1):
    """Crop an array from the padded grid down to the user's grid."""
    if pad_grid is user_grid:
        return arr
    col_off = round((user_grid["transform"].xoff - pad_grid["transform"].xoff) / RESOLUTION)
    row_off = round((pad_grid["transform"].yoff - user_grid["transform"].yoff) / RESOLUTION)
    out_h, out_w = user_grid["shape"]
    if channel_axis == 0:
        return arr[:, row_off:row_off + out_h, col_off:col_off + out_w]
    return arr[row_off:row_off + out_h, col_off:col_off + out_w, ...]


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
            start_date = st.date_input("Start", date(2023, 6, 1), min_value=date(2017, 1, 1))
        with col_d2:
            end_date = st.date_input("End", date(2023, 8, 31), max_value=date(2025, 12, 31))
        years = sorted(set(range(start_date.year, end_date.year + 1)))
        custom_dates = (str(start_date), str(end_date))

    min_coverage = st.slider("Min scene coverage (%)", 50, 100, 100, step=5)
    max_cloud = st.slider("Max cloud cover (%)", 5, 50, 20, step=5)
    save_per_timestamp = st.toggle("Save per-timestamp embeddings", value=True,
                                    help="Disable to only output the annual average (much smaller download)")
    save_per_timestamp_input = st.toggle("Save per-timestamp input", value=False,
                                          help="Also save the raw S2/S1 scene data used for each timestamp (large: adds raw bands per scene)")

    st.divider()

    # Bbox display
    if "bbox" in st.session_state and st.session_state.bbox:
        bbox = st.session_state.bbox
        n_years = len(years)
        w_km, h_km, total_mb = estimate_size(
            bbox, n_years=n_years,
            save_per_timestamp=save_per_timestamp,
            save_per_timestamp_input=save_per_timestamp_input,
        )
        st.metric("Area", f"{w_km} × {h_km} km")
        label = f"{total_mb} MB" + (f" ({n_years} yr)" if n_years > 1 else "")
        if total_mb <= MAX_OUTPUT_MB:
            st.metric("Est. output", label, delta="OK", delta_color="normal")
        else:
            st.metric("Est. output", label, delta=f">{MAX_OUTPUT_MB} MB!", delta_color="inverse")
        _, was_padded = expand_to_min(bbox)
        if was_padded:
            st.info(f"Small area — will be padded internally to {MIN_SIDE_KM} km per side, output cropped back.")
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

    st.divider()
    st.caption(
        "[GitHub](https://github.com/asterisk-labs/beta-earth) · "
        "[Google Satellite Embedding (AEF)](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)"
    )



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

# CartoDB Positron — clean minimal basemap (no API key needed)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="CartoDB",
    name="Light",
    overlay=False,
    show=True,
).add_to(m)

# Draw control for bbox
folium.plugins.Draw(
    draw_options={
        "polyline": False, "polygon": False, "circle": False,
        "circlemarker": False, "marker": False,
        "rectangle": {"shapeOptions": {"color": "#c4ffc2", "weight": 3, "fillOpacity": 0.1}},
    },
    edit_options={"edit": False},
).add_to(m)

# Persist the drawn bbox as a visible rectangle + fit view, OR auto-activate
# the draw tool if no bbox has been set yet.
if "bbox" in st.session_state and st.session_state.bbox:
    sbbox = st.session_state.bbox
    folium.Rectangle(
        bounds=[[sbbox[1], sbbox[0]], [sbbox[3], sbbox[2]]],
        color="#c4ffc2", weight=3, fill=True, fill_opacity=0.1,
    ).add_to(m)
    m.fit_bounds([[sbbox[1], sbbox[0]], [sbbox[3], sbbox[2]]])
else:
    from branca.element import MacroElement
    from jinja2 import Template

    class AutoDrawRectangle(MacroElement):
        _template = Template("""
        {% macro script(this, kwargs) %}
        setTimeout(function() {
            var btn = document.querySelector('.leaflet-draw-draw-rectangle');
            if (btn) btn.click();
        }, 300);
        {% endmacro %}
        """)

    m.add_child(AutoDrawRectangle())

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
    n_years = len(years)
    w_km, h_km, total_mb = estimate_size(
        bbox, n_years=n_years,
        save_per_timestamp=save_per_timestamp,
        save_per_timestamp_input=save_per_timestamp_input,
    )
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

        user_grid = compute_grid(bbox)
        pad_bbox, was_padded = expand_to_min(bbox)
        grid = compute_grid(pad_bbox) if was_padded else user_grid
        h, w = grid["shape"]
        out_h, out_w = user_grid["shape"]

        run_id = uuid.uuid4().hex[:8]
        output_dir = Path(tempfile.mkdtemp()) / f"betaearth_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # DEM (shared across years) — download at padded size, save cropped
        progress.progress(8, text="Downloading DEM...")
        dem = download_dem(grid)
        if save_per_timestamp_input:
            dem_out = crop_to_user(dem, grid, user_grid, channel_axis=0)
            write_geotiff(dem_out.astype(np.float32), user_grid, output_dir / "dem.tif")
            # DEM preview (from cropped array)
            from PIL import Image
            d = dem_out[0].astype(np.float32)
            lo, hi = np.percentile(d[np.isfinite(d)], [2, 98])
            d_norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            Image.fromarray((d_norm * 255).astype(np.uint8)).save(output_dir / "dem_preview.png")

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
            if custom_dates:
                import pystac_client, planetary_computer
                from datetime import date as _date
                cd_start = _date.fromisoformat(custom_dates[0])
                cd_end = _date.fromisoformat(custom_dates[1])
                yr_start = max(cd_start, _date(year, 1, 1))
                yr_end = min(cd_end, _date(year, 12, 31))
                catalog = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=planetary_computer.sign_inplace,
                )
                s2_items = list(catalog.search(
                    collections=["sentinel-2-l2a"], bbox=list(bbox),
                    datetime=f"{yr_start}/{yr_end}",
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    max_items=200,
                ).items())
                s1_items = list(catalog.search(
                    collections=["sentinel-1-rtc"], bbox=list(bbox),
                    datetime=f"{yr_start}/{yr_end}",
                    max_items=200,
                ).items())
            else:
                s2_items = _search_stac(bbox, year, "sentinel-2-l2a", max_cloud=max_cloud)
                s1_items = _search_stac(bbox, year, "sentinel-1-rtc")
            s2_items = _seasonal_select(s2_items, max_per_quarter=6)
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
                    emb_out = crop_to_user(emb, grid, user_grid, channel_axis=-1)
                    write_geotiff(emb_out.astype(np.float32), user_grid, ts_dir / "embedding.tif")
                if save_per_timestamp_input:
                    # data is band-first: (C, H, W) — crop band-first
                    data_out = crop_to_user(data, grid, user_grid, channel_axis=0)
                    write_geotiff(data_out.astype(np.float32), user_grid, ts_dir / "input.tif")
                    # RGB preview (from cropped input)
                    from PIL import Image
                    if sensor == "S2":
                        # S2 band order: [B02, B03, B04, B08, B05, B06, B07, B11, B12]
                        # RGB = B04, B03, B02 → indices 2, 1, 0
                        rgb = np.stack([data_out[2], data_out[1], data_out[0]], axis=-1).astype(np.float32)
                        rgb = np.clip(rgb / 3000.0, 0, 1) ** (1/2.2)
                    else:
                        # S1: VV, VH → composite with ratio for third channel
                        vv = 10 * np.log10(np.clip(data_out[0], 1e-6, None))
                        vh = 10 * np.log10(np.clip(data_out[1], 1e-6, None))
                        ratio = vv - vh
                        def _norm(x):
                            lo, hi = np.percentile(x[np.isfinite(x)], [2, 98])
                            return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
                        rgb = np.stack([_norm(vv), _norm(vh), _norm(ratio)], axis=-1)
                    Image.fromarray((rgb * 255).astype(np.uint8)).save(ts_dir / "preview_rgb.png")
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
            avg_out = crop_to_user(avg, grid, user_grid, channel_axis=-1)
            write_geotiff(avg_out, user_grid, output_dir / f"{year}.tif")

            # PCA previews (fit on cropped annual, apply same basis to timestamps)
            progress.progress(yr_hi - 2, text=f"{yr_label}PCA previews...")
            pca_state = fit_pca(avg_out)
            annual_preview = output_dir / f"{year}_preview_pca.png"
            write_pca_preview(avg_out, annual_preview, pca_state=pca_state)
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
                rgb_png = ts_dir / "preview_rgb.png"
                if rgb_png.exists():
                    all_previews.append((str(rgb_png), f"{year}/{ts_dir.name} (RGB input)"))

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

        # Log request metadata asynchronously (fire-and-forget)
        area_km2 = w_km * h_km
        log_request(
            bbox=bbox,
            area_km2=area_km2,
            years=years,
            time_mode=time_mode,
            save_per_timestamp=save_per_timestamp,
            save_per_timestamp_input=save_per_timestamp_input,
        )

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
