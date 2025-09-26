# -----------------------------
# Hybrid Sentiment — Fashion & Cosmetics (Fast vs Full)
# -----------------------------
import os
import io
import json
import subprocess
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional libs you already use
import matplotlib.pyplot as plt
import altair as alt

# ------------- Page config & styles -------------
st.set_page_config(
    page_title="Hybrid Sentiment — Fashion & Cosmetics",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle theme accents
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#fff8fb, #fff); }
      .metric small { color:#666; }
      .badge { 
        display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;
        background:#f0f0f5;margin-right:6px;border:1px solid #e7e7ef
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------- Constants -------------
DEMO_PATH_DEFAULT = "data/processed/combined_demo.csv"  # small sample kept in repo

# ------------- Helpers: data loading -------------
@st.cache_data(show_spinner=False)
def ensure_full_csv() -> Tuple[Optional[str], str]:
    """
    Return (path_to_full_csv, source_label).

    Priority:
      1) Local file (your laptop): st.secrets.FULL_LOCAL_PATH or data/processed/combined_hybrid.csv
      2) Google Drive (Streamlit Cloud): download by ID to st.secrets.FULL_CACHE_PATH or /tmp/combined_hybrid.csv
      3) None (not available)
    """
    local_path = st.secrets.get("FULL_LOCAL_PATH", "data/processed/combined_hybrid.csv")
    cache_path = st.secrets.get("FULL_CACHE_PATH", "/tmp/combined_hybrid.csv")
    gdrive_id = st.secrets.get("GDRIVE_ID", "")

    # 1) Local full file for local runs
    try:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            return local_path, "Local file"
    except Exception:
        pass

    # 2) Try Google Drive download (only if GDRIVE_ID provided)
    if gdrive_id:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # download only if missing/suspiciously small
            if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 1024:
                # use gdown cli (installed via requirements)
                cmd = ["gdown", "--id", gdrive_id, "-O", cache_path, "--fuzzy"]
                subprocess.check_call(cmd)
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
                return cache_path, "Google Drive (cached)"
        except Exception as e:
            st.error(f"Full dataset download failed: {e}")

    # 3) Not available
    return None, "Not available"

@st.cache_data(show_spinner=False)
def read_csv_smart(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """One place to tune read_csv options. Set dtype/usecols here if needed."""
    return pd.read_csv(path, low_memory=False, usecols=usecols)

def load_dataset(mode: str) -> Tuple[pd.DataFrame, str]:
    """Load dataset based on mode; returns (df, source_label)."""
    if mode == "Fast":
        if not os.path.exists(DEMO_PATH_DEFAULT):
            st.error("Demo dataset missing: data/processed/combined_demo.csv")
            st.stop()
        return read_csv_smart(DEMO_PATH_DEFAULT), "Demo sample (repo)"

    # Full
    st.info("Loading full dataset… (local if present; else download from Google Drive)")
    full_path, origin = ensure_full_csv()
    if not full_path:
        st.warning("Full dataset not available. Falling back to demo sample.")
        return read_csv_smart(DEMO_PATH_DEFAULT), "Fallback demo sample"
    return read_csv_smart(full_path), origin

# ------------- Sidebar controls -------------
st.sidebar.header("Data & Controls")

# Mode
mode = st.sidebar.radio("Mode", ["Fast", "Full"], index=0)

# (Optional) upload to override any mode
uploaded = st.sidebar.file_uploader("Upload CSV to analyze (optional)", type=["csv"])

# ------------- Load data -------------
if uploaded is not None:
    df = pd.read_csv(uploaded, low_memory=False)
    data_origin = "User upload"
else:
    df, data_origin = load_dataset(mode)

# ------------- Data source badge -------------
source_icon = {
    "Demo sample (repo)": "📦",
    "Local file": "💻",
    "Google Drive (cached)": "☁️",
    "Fallback demo sample": "📦",
    "User upload": "📤",
}
st.markdown(
    f"**Data source:** <span class='badge'>{source_icon.get(data_origin,'ℹ️')} {data_origin}</span> "
    f"<span class='badge'>Mode: {mode}</span> "
    f"<span class='badge'>Rows: {len(df):,}</span>",
    unsafe_allow_html=True,
)

# ------------- Column detection (robust to different schemas) -------------
# Try to align with your known schema keys; gracefully degrade if missing
name_map = {
    "text": ["text", "review_text", "content", "body"],
    "brand": ["brand_name", "brand", "Brand"],
    "price": ["price", "mrp", "Price"],
    "rating": ["rating", "review_rating", "product_rating"],
    "label": ["hybrid_label", "review_label", "target"],
    "product_id": ["product_id", "asin", "parent_asin"],
    "product_title": ["product_title", "title"],
    "timestamp": ["timestamp", "review_date", "date"],
    "verified": ["verified_purchases", "is_a_buyer"],
}

def pick(col_aliases: List[str]) -> Optional[str]:
    for c in col_aliases:
        if c in df.columns:
            return c
    return None

COL_TEXT = pick(name_map["text"])
COL_BRAND = pick(name_map["brand"])
COL_PRICE = pick(name_map["price"])
COL_RATING = pick(name_map["rating"])
COL_LABEL = pick(name_map["label"])
COL_PID = pick(name_map["product_id"])
COL_PTITLE = pick(name_map["product_title"])
COL_TIME = pick(name_map["timestamp"])
COL_VER = pick(name_map["verified"])

# ------------- Filters (only render if the columns exist) -------------
with st.sidebar.expander("Interactive filters", expanded=True):
    # Brand filter
    if COL_BRAND and df[COL_BRAND].notna().any():
        brands = sorted([str(x) for x in df[COL_BRAND].dropna().unique().tolist()])[:3000]
        sel_brands = st.multiselect("Filter by brand", options=brands, default=[])
    else:
        sel_brands = []
    # Price range
    if COL_PRICE and df[COL_PRICE].notna().any():
        try:
            prices = pd.to_numeric(df[COL_PRICE], errors="coerce").dropna()
            pmin, pmax = float(np.nanmin(prices)), float(np.nanmax(prices))
            price_range = st.slider("Price range", min_value=float(pmin), max_value=float(pmax),
                                    value=(float(pmin), float(pmax)))
        except Exception:
            price_range = None
    else:
        price_range = None
    # Verified
    ver_choice = "All"
    if COL_VER and df[COL_VER].notna().any():
        ver_choice = st.selectbox("Verified purchase filter", ["All", "Verified only", "Unverified only"])

# Apply filters
filtered = df.copy()
if sel_brands and COL_BRAND:
    filtered = filtered[filtered[COL_BRAND].astype(str).isin(sel_brands)]
if price_range and COL_PRICE:
    filtered = filtered[(pd.to_numeric(filtered[COL_PRICE], errors="coerce") >= price_range[0]) &
                        (pd.to_numeric(filtered[COL_PRICE], errors="coerce") <= price_range[1])]
if ver_choice != "All" and COL_VER:
    flag = (filtered[COL_VER].astype(str).str.lower().isin(["true", "1", "yes"]))
    filtered = filtered[flag] if ver_choice == "Verified only" else filtered[~flag]

st.write(f"**Filtered rows:** {len(filtered):,}")

# ------------- Analytics Dashboard -------------
st.subheader("Analytics Dashboard (filtered)")

# Label counts
if COL_LABEL and COL_LABEL in filtered.columns:
    counts = filtered[COL_LABEL].astype(str).value_counts().reset_index()
    counts.columns = ["label", "count"]
    st.write("Label counts:")
    st.dataframe(counts, use_container_width=True)

    # Altair bar (guard for empty)
    if len(counts) > 0:
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("label:N", sort="-y"), y="count:Q", tooltip=["label", "count"]
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No label column found (expected one of: hybrid_label / review_label / target).")

# Keyword-ish terms (very lightweight example on sample)
with st.expander("Top keywords by sentiment (lightweight)", expanded=False):
    if COL_TEXT and COL_LABEL and len(filtered) > 0:
        # super small sample to keep things quick on Cloud
        sample_n = min(5000, len(filtered))
        tmp = filtered[[COL_TEXT, COL_LABEL]].dropna().sample(n=sample_n, random_state=42)
        # simple token frequency by label (demo only)
        def top_terms(dfpart: pd.DataFrame, k=15):
            toks = " ".join(dfpart[COL_TEXT].astype(str).tolist()).lower().split()
            s = pd.Series(toks)
            s = s[s.str.len() > 2]
            return s.value_counts().head(k).reset_index().values.tolist()

        labels_present = tmp[COL_LABEL].astype(str).unique().tolist()
        cols = st.columns(len(labels_present))
        for i, lab in enumerate(labels_present):
            terms = top_terms(tmp[tmp[COL_LABEL].astype(str) == lab])
            with cols[i]:
                st.write(f"**{lab} — top terms**")
                if terms:
                    st.table(pd.DataFrame(terms, columns=["term", "count"]))
                else:
                    st.write("—")
    else:
        st.write("Need text + label columns to compute top terms.")

# Product-level rollups
st.subheader("Product-level rollups (filtered)")
if COL_PID or COL_PTITLE:
    group_cols = [c for c in [COL_PID, COL_PTITLE, COL_BRAND] if c]
    roll = filtered.copy()
    # Coerce rating numeric
    if COL_RATING and COL_RATING in roll.columns:
        roll["_rating_num"] = pd.to_numeric(roll[COL_RATING], errors="coerce")
    else:
        roll["_rating_num"] = np.nan
    agg = {
        "_rating_num": ["mean", "count"],
    }
    if COL_LABEL:
        agg[COL_LABEL] = [lambda s: (s.astype(str) == "positive").mean(),
                          lambda s: (s.astype(str) == "negative").mean()]
    prod = roll.groupby(group_cols).agg(agg)
    # Flatten columns
    prod.columns = ["avg_rating", "review_count", "positive_rate", "negative_rate"]
    prod = prod.reset_index().sort_values("review_count", ascending=False).head(200)
    st.dataframe(prod, use_container_width=True)

    # Download button
    csv_bytes = prod.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download product rollup CSV",
        data=csv_bytes,
        file_name="product_rollup.csv",
        mime="text/csv",
    )
else:
    st.info("No product columns found (expected product_id / asin / parent_asin and/or product_title).")

# ------------- Single review prediction demo (optional placeholder) -------------
with st.expander("Single review prediction (demo)", expanded=False):
    txt = st.text_area("Paste review text", placeholder="E.g., Love this lipstick — color pops and lasts!")
    if txt:
        # Placeholder: simple heuristic if you don't have model loaded here
        score = (txt.lower().count("love") + txt.lower().count("great")) - (txt.lower().count("bad") + txt.lower().count("terrible"))
        label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
        st.write(f"**Predicted (demo):** {label}")

# ------------- User manual -------------
with st.expander("📘 User Manual (what’s what)", expanded=False):
    st.markdown(
        """
**Modes**
- **Fast**: uses a small demo sample (`data/processed/combined_demo.csv`) committed in the repo → fast & ideal for Cloud.
- **Full**: 
  - **Locally** loads your `data/processed/combined_hybrid.csv`.
  - **On Streamlit Cloud** downloads the same file from Google Drive (first run), caches in `/tmp/`, then uses it.

**Upload**
- You can optionally upload a CSV; if you do, the app analyzes your uploaded file instead of the built-in datasets.

**Expected columns (auto-detected)**  
We auto-detect common names. Best effort mapping for:
- Text: `text`, `review_text`, `content`, `body`
- Label: `hybrid_label`, `review_label`, `target`
- Rating: `rating`, `review_rating`, `product_rating`
- Brand: `brand_name`, `brand`
- Product: `product_id`, `asin`, `parent_asin`, `product_title` / `title`
- Price: `price`, `mrp`
- Verified: `verified_purchases`, `is_a_buyer`
- Time: `timestamp`, `review_date`, `date`

**Downloads**
- Use the "Download product rollup CSV" button to export aggregated product metrics.

If something doesn’t appear (filters or charts), it likely means the required column isn’t present in the current dataset (upload or sample).
"""
    )
