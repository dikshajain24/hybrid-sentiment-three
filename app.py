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
import altair as alt
import matplotlib.pyplot as plt

# ---------- Page config & styles ----------
st.set_page_config(
    page_title="Hybrid Sentiment — Fashion & Cosmetics",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#fff8fb, #fff); }
      .app-hero h1 { font-size: 40px; margin: 0; }
      .app-hero small { color:#5f5f6e; }
      .badge { 
        display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;
        background:#f5f5fb;margin-right:6px;border:1px solid #e8e8f7
      }
      .section-title{margin-top:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Constants ----------
DEMO_PATH_DEFAULT = "data/processed/combined_demo.csv"  # small sample kept in repo

# ---------- Helpers: data loading ----------
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
            if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 1024:
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
    return pd.read_csv(path, low_memory=False, usecols=usecols)

def load_dataset(mode: str) -> Tuple[pd.DataFrame, str]:
    if mode == "Fast":
        if not os.path.exists(DEMO_PATH_DEFAULT):
            st.error("Demo dataset missing: data/processed/combined_demo.csv")
            st.stop()
        return read_csv_smart(DEMO_PATH_DEFAULT), "Demo sample (repo)"

    st.info("Loading full dataset… (local if present; else Google Drive)")
    full_path, origin = ensure_full_csv()
    if not full_path:
        st.warning("Full dataset not available. Falling back to demo sample.")
        return read_csv_smart(DEMO_PATH_DEFAULT), "Fallback demo sample"
    return read_csv_smart(full_path), origin

# ---------- Sidebar ----------
st.sidebar.header("Data & Controls")

# Manual / Help (restored)
with st.sidebar.expander("❓ User Manual — click to open", expanded=False):
    st.markdown(
        """
**Modes**
- **Fast** → uses a small demo sample (`data/processed/combined_demo.csv`) in the repo (best for Cloud).
- **Full** → uses your full dataset  
  - **Locally** from `data/processed/combined_hybrid.csv`  
  - **On Streamlit Cloud** downloaded from Google Drive and cached in `/tmp/`.

**Upload (optional)**
- Upload a CSV to override and analyze your file directly.

**Columns (auto-detected)**  
We look for the following aliases:
- Text: `text`, `review_text`, `content`, `body`
- Label: `hybrid_label`, `review_label`, `target`
- Rating: `rating`, `review_rating`, `product_rating`
- Brand: `brand_name`, `brand`
- Product: `product_id`, `asin`, `parent_asin`, `product_title` / `title`
- Price: `price`, `mrp`
- Verified: `verified_purchases`, `is_a_buyer`
- Time: `timestamp`, `review_date`, `date`
        """
    )

mode = st.sidebar.radio("Mode", ["Fast", "Full"], index=0)
uploaded = st.sidebar.file_uploader("Upload CSV to analyze (optional)", type=["csv"])

# ---------- Load data ----------
if uploaded is not None:
    df = pd.read_csv(uploaded, low_memory=False)
    data_origin = "User upload"
else:
    df, data_origin = load_dataset(mode)

# ---------- Header ----------
st.markdown(
    """
<div class="app-hero">
  <h1>💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics</h1>
  <small>Analyze brand sentiment, explore trends, and surface top keywords across your datasets.</small>
</div>
""",
    unsafe_allow_html=True,
)

# Data-source badge
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

# ---------- Column detection ----------
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

def pick(df_: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    for c in aliases:
        if c in df_.columns:
            return c
    return None

COL_TEXT = pick(df, name_map["text"])
COL_BRAND = pick(df, name_map["brand"])
COL_PRICE = pick(df, name_map["price"])
COL_RATING = pick(df, name_map["rating"])
COL_LABEL = pick(df, name_map["label"])
COL_PID = pick(df, name_map["product_id"])
COL_PTITLE = pick(df, name_map["product_title"])
COL_TIME = pick(df, name_map["timestamp"])
COL_VER = pick(df, name_map["verified"])

# ---------- Filters ----------
with st.sidebar.expander("Interactive filters", expanded=True):
    # Brand
    if COL_BRAND and df[COL_BRAND].notna().any():
        brands = sorted([str(x) for x in df[COL_BRAND].dropna().unique().tolist()])[:3000]
        sel_brands = st.multiselect("Filter by brand", options=brands, default=[])
    else:
        sel_brands = []

    # Price
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

# ---------- Analytics ----------
st.subheader("Analytics Dashboard (filtered)")

# Label counts
if COL_LABEL and COL_LABEL in filtered.columns:
    counts = filtered[COL_LABEL].astype(str).value_counts().reset_index()
    counts.columns = ["label", "count"]
    st.write("Label counts:")
    st.dataframe(counts, use_container_width=True)

    if len(counts) > 0:
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("label:N", sort="-y"),
            y=alt.Y("count:Q"),
            tooltip=["label", "count"]
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No label column found (expected one of: hybrid_label / review_label / target).")

# ---------- Top keywords (fixed sampling) ----------
with st.expander("Top keywords by sentiment (filtered sample)", expanded=False):
    if COL_TEXT and COL_LABEL and len(filtered) > 0:
        sample_n = st.slider("Sample size for keywords", 200, 5000, 2000, 100)
        base = filtered[[COL_TEXT, COL_LABEL]].dropna()
        avail = len(base)
        if avail == 0:
            st.write("No text samples available for keyword extraction.")
        else:
            n = min(sample_n, avail)
            tmp = base.sample(n=n, random_state=42)
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

# ---------- Product-level rollups & download ----------
st.subheader("Product-level rollups (filtered)")
if COL_PID or COL_PTITLE:
    group_cols = [c for c in [COL_PID, COL_PTITLE, COL_BRAND] if c]
    roll = filtered.copy()
    if COL_RATING and COL_RATING in roll.columns:
        roll["_rating_num"] = pd.to_numeric(roll[COL_RATING], errors="coerce")
    else:
        roll["_rating_num"] = np.nan
    agg = {"_rating_num": ["mean", "count"]}
    if COL_LABEL:
        agg[COL_LABEL] = [
            lambda s: float((s.astype(str) == "positive").mean()),
            lambda s: float((s.astype(str) == "negative").mean()),
        ]
    prod = roll.groupby(group_cols).agg(agg)
    prod.columns = ["avg_rating", "n_reviews", "positive_share", "negative_share"] if COL_LABEL else ["avg_rating", "n_reviews"]
    prod = prod.reset_index().sort_values("n_reviews", ascending=False).head(500)
    st.dataframe(prod, use_container_width=True)

    csv_bytes = prod.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download product rollups CSV",
        data=csv_bytes,
        file_name="product_rollups.csv",
        mime="text/csv",
    )
else:
    st.info("No product columns found (expected product_id / asin / parent_asin and/or product_title).")

# ---------- Single review prediction (simple demo) ----------
with st.expander("Single review prediction (demo)", expanded=False):
    txt = st.text_area("Paste review text", placeholder="E.g., 'Love this lipstick — color pops and lasts!'")
    if txt:
        score = (txt.lower().count("love") + txt.lower().count("great")) - (txt.lower().count("bad") + txt.lower().count("terrible"))
        label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
        st.write(f"**Predicted (demo):** {label}")
