import os
import re
import string
import subprocess
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Optional/extra viz
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

# -----------------------------
# Page config & basic styles
# -----------------------------
st.set_page_config(
    page_title="💄✨ Hybrid Sentiment — Fashion & Cosmetics",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#fff8fb, #fff); }
      .app-hero h1 { font-size: 40px; margin: 0 0 6px 0; }
      .app-hero small { color:#5f5f6e; }
      .badge { display:inline-block;padding:4px 10px;border-radius:999px;
               font-size:12px;background:#f5f5fb;margin-right:6px;border:1px solid #e8e8f7 }
      .section-title{margin-top:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Constants
# -----------------------------
DEMO_PATH = "data/processed/combined_demo.csv"  # small repo sample

# column alias map (auto-detect)
ALIASES = {
    "text": ["review_text", "text", "content", "body", "title"],
    "label": ["hybrid_label", "review_label", "target", "label"],
    "brand": ["brand_name", "brand", "Brand"],
    "price": ["price", "mrp", "Price"],
    "rating": ["rating", "review_rating", "product_rating"],
    "product_id": ["product_id", "asin", "parent_asin"],
    "product_title": ["product_title", "title"],
    "timestamp": ["timestamp", "review_date", "date"],
    "verified": ["verified_purchases", "is_a_buyer", "verified_purchase"],
}

# -----------------------------
# Utilities
# -----------------------------
def pick_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for c in names:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def clean_tokens(s: str) -> list[str]:
    s = str(s).lower()
    s = re.sub(r"http\\S+|www\\.\\S+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\\s+", " ", s).strip()
    return [t for t in s.split(" ") if t and not t.isdigit() and len(t) > 2]

def safe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) == 0:
        return df.iloc[0:0]
    n = min(n, len(df))
    if n <= 0:
        return df.iloc[0:0]
    return df.sample(n=n, random_state=seed)

# -----------------------------
# Data loaders (Fast / Full)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_demo() -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(DEMO_PATH):
        return pd.DataFrame(), "Demo sample not found"
    return read_csv_cached(DEMO_PATH), "Demo sample (repo)"

@st.cache_data(show_spinner=True)
def load_full_local(path: str) -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(path):
        return pd.DataFrame(), f"Local path not found: {path}"
    return read_csv_cached(path), f"Full dataset (local): {os.path.basename(path)}"

@st.cache_data(show_spinner=True)
def load_full_drive(file_id: str) -> Tuple[pd.DataFrame, str]:
    if not file_id.strip():
        return pd.DataFrame(), "No Drive file ID provided"
    try:
        # ensure gdown is present in the runtime (requirements.txt has gdown)
        cmd = ["gdown", "--id", file_id, "-O", "/tmp/full.csv", "--fuzzy"]
        subprocess.check_call(cmd)
        return read_csv_cached("/tmp/full.csv"), "Full dataset (Google Drive)"
    except Exception as e:
        return pd.DataFrame(), f"Drive download failed: {e}"

# -----------------------------
# Header + Sidebar
# -----------------------------
st.markdown(
    """
<div class="app-hero">
  <h1>💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics</h1>
  <small>Analyze brand sentiment, explore trends, and surface top keywords across your datasets.</small>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("❓ User Manual — click to open", expanded=False):
    st.markdown(
        """
**Modes**
- **Fast** → uses a small demo sample (`data/processed/combined_demo.csv`) — best for Cloud.
- **Full** → analyze full CSV from a **local path** or **Google Drive** ID.

**Upload (optional)** → analyze your own CSV (overrides Fast/Full).

**Auto-detected columns**  
Text · Label · Brand · Product · Price · Rating · Verified · Timestamp  
(We match common names; use the same names as your processed files.)

**Tips**  
- If a section says “no data”, clear filters or switch modes.  
- Wordclouds need `wordcloud` (already in requirements).  
- Full via Drive needs `gdown` (already in requirements).
"""
    )

st.sidebar.header("Data & Controls")

mode = st.sidebar.radio("Mode", ["Fast", "Full"], index=0)
uploaded = st.sidebar.file_uploader("Upload CSV to analyze (optional)", type=["csv"])

drive_id = st.sidebar.text_input(
    "Google Drive file ID (Full mode, optional)",
    help="From a shared link: https://drive.google.com/file/d/**FILE_ID**/view"
)
local_path = st.sidebar.text_input(
    "Local full CSV path (optional)",
    help="Example: C:/Users/you/Desktop/combined_hybrid.csv"
)

# -----------------------------
# Load selected source
# -----------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded, low_memory=False)
    source = "User upload"
else:
    if mode == "Fast":
        df, source = load_demo()
    else:
        df, source = (pd.DataFrame(), "Full (no file provided)")
        if local_path.strip():
            df, source = load_full_local(local_path.strip())
        elif drive_id.strip():
            df, source = load_full_drive(drive_id.strip())

if df.empty:
    st.info("No data to show. Upload a CSV or provide Full dataset details, or switch to **Fast**.")
    st.stop()

# -----------------------------
# Column detection
# -----------------------------
COL_TEXT   = pick_column(df, ALIASES["text"])
COL_LABEL  = pick_column(df, ALIASES["label"])
COL_BRAND  = pick_column(df, ALIASES["brand"])
COL_PRICE  = pick_column(df, ALIASES["price"])
COL_RATING = pick_column(df, ALIASES["rating"])
COL_PID    = pick_column(df, ALIASES["product_id"])
COL_PTITLE = pick_column(df, ALIASES["product_title"])
COL_TIME   = pick_column(df, ALIASES["timestamp"])
COL_VER    = pick_column(df, ALIASES["verified"])

missing_required = [c for c in [("text", COL_TEXT), ("label", COL_LABEL)] if c[1] is None]
if missing_required:
    st.error(f"Required columns not found: {[m[0] for m in missing_required]}. "
             f"Found columns: {list(df.columns)[:20]} …")
    st.stop()

# Timestamp parse (if present)
if COL_TIME:
    try:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    except Exception:
        pass

# Source badges
st.markdown(
    f"**Data source:** <span class='badge'>{source}</span> "
    f"<span class='badge'>Mode: {mode}</span> "
    f"<span class='badge'>Rows: {len(df):,}</span>",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar.expander("Interactive filters", expanded=True):
    # Brand
    if COL_BRAND and df[COL_BRAND].notna().any():
        brands = sorted([str(x) for x in df[COL_BRAND].dropna().unique().tolist()])[:3000]
        sel_brands = st.multiselect("Filter by brand", options=brands, default=[])
    else:
        sel_brands = []

    # Price
    if COL_PRICE:
        prices = pd.to_numeric(df[COL_PRICE], errors="coerce")
        if prices.notna().any():
            lo, hi = float(prices.min()), float(prices.max())
            price_range = st.slider("Price range", min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi)))
        else:
            price_range = None
            st.caption("⚠️ Price column has no numeric values.")
    else:
        price_range = None
        st.caption("⚠️ Price column not available.")

    # Verified
    ver_choice = "All"
    if COL_VER and df[COL_VER].notna().any():
        ver_choice = st.selectbox("Verified purchase filter", ["All", "Verified only", "Unverified only"])

# Apply filters
filtered = df.copy()
if sel_brands and COL_BRAND:
    filtered = filtered[filtered[COL_BRAND].astype(str).isin(sel_brands)]
if price_range and COL_PRICE:
    pr = pd.to_numeric(filtered[COL_PRICE], errors="coerce")
    filtered = filtered[(pr >= price_range[0]) & (pr <= price_range[1])]
if ver_choice != "All" and COL_VER:
    flag = filtered[COL_VER].astype(str).str.lower().isin(["true", "1", "yes"])
    filtered = filtered[flag] if ver_choice == "Verified only" else filtered[~flag]

st.caption(f"Filtered rows: **{len(filtered):,}**")

# -----------------------------
# Analytics: label counts
# -----------------------------
st.subheader("Analytics Dashboard (filtered)")
if COL_LABEL in filtered.columns and len(filtered) > 0:
    counts = filtered[COL_LABEL].astype(str).value_counts().reset_index()
    counts.columns = ["label", "count"]
    st.dataframe(counts, use_container_width=True)
    if len(counts) > 0:
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("label:N", sort="-y"), y=alt.Y("count:Q"),
            tooltip=["label", "count"]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No label counts to display.")

# -----------------------------
# Top keywords by sentiment (lightweight)
# -----------------------------
st.markdown("### Top keywords by sentiment (filtered sample)")
if COL_TEXT and COL_LABEL and len(filtered) > 0:
    sample_n = st.slider("Sample size for keywords", 200, 10000, 2000, 100)
    base = filtered[[COL_TEXT, COL_LABEL]].dropna()
    avail = len(base)
    if avail == 0:
        st.write("No text samples available for keyword extraction.")
    else:
        tmp = safe_sample(base, n=sample_n, seed=42)

        def top_terms(dfpart: pd.DataFrame, k=20):
            toks = []
            for t in dfpart[COL_TEXT].astype(str):
                toks.extend(clean_tokens(t))
            if not toks:
                return [], ""
            s = pd.Series(toks)
            s = s[s.str.len() > 2]
            vc = s.value_counts().head(k)
            return list(zip(vc.index.tolist(), vc.values.tolist())), " ".join(s.tolist())

        show_clouds = st.checkbox("Show wordclouds", value=True and HAS_WORDCLOUD)
        if show_clouds and not HAS_WORDCLOUD:
            st.caption("Install `wordcloud` to enable clouds (already in requirements.txt).")

        labels_present = tmp[COL_LABEL].astype(str).unique().tolist()
        cols = st.columns(len(labels_present))
        for i, lab in enumerate(labels_present):
            subset = tmp[tmp[COL_LABEL].astype(str) == lab]
            terms, all_text = top_terms(subset)
            with cols[i]:
                st.write(f"**{lab} — top terms**")
                if terms:
                    st.table(pd.DataFrame(terms, columns=["term", "count"]))
                    if show_clouds and HAS_WORDCLOUD:
                        wc = WordCloud(width=700, height=360, background_color="white").generate(all_text)
                        st.image(wc.to_array(), use_column_width=True)
                else:
                    st.write("—")
else:
    st.write("Need text + label columns for keywords.")

# -----------------------------
# Product-level rollups + download
# -----------------------------
st.subheader("Product-level rollups (filtered)")
if (COL_PID or COL_PTITLE) and len(filtered) > 0:
    group_cols = [c for c in [COL_PID, COL_PTITLE, COL_BRAND] if c]
    roll = filtered.copy()

    # rating
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

    # flatten
    if COL_LABEL:
        prod.columns = ["avg_rating", "n_reviews", "positive_share", "negative_share"]
    else:
        prod.columns = ["avg_rating", "n_reviews"]
        prod["positive_share"] = np.nan
        prod["negative_share"] = np.nan

    prod = prod.reset_index()

    st.dataframe(prod.sort_values("n_reviews", ascending=False).head(500), use_container_width=True)

    st.download_button(
        "Download product rollups CSV",
        data=prod.to_csv(index=False).encode("utf-8"),
        file_name="product_rollups.csv",
        mime="text/csv",
    )

    # -------- Top-N products chart (after rollups)
    st.subheader("Top products by selected metric")
    metric = st.selectbox("Sort top products by", ["positive_share", "avg_rating", "n_reviews"], index=0)
    top_n = st.slider("Top N products", 5, 50, 20, 1)

    name_col = COL_PTITLE or COL_PID or (COL_BRAND if COL_BRAND in prod.columns else None)
    if not name_col:
        name_col = prod.columns[0]

    top_df = prod[[name_col, "positive_share", "avg_rating", "n_reviews"]].copy()
    if metric != "n_reviews":
        top_df = top_df.dropna(subset=[metric])
    top_df = top_df.sort_values(metric, ascending=False).head(top_n)

    if len(top_df) == 0:
        st.info("No products for this metric/filter combination.")
    else:
        chart_top = alt.Chart(top_df).mark_bar().encode(
            x=alt.X(f"{metric}:Q"),
            y=alt.Y(f"{name_col}:N", sort="-x"),
            tooltip=[name_col, metric, "avg_rating", "n_reviews", "positive_share"]
        ).properties(height=max(240, 22*len(top_df)))
        st.altair_chart(chart_top, use_container_width=True)

        st.download_button(
            "Download top-N products CSV",
            data=top_df.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{top_n}_{metric}.csv",
            mime="text/csv",
        )
else:
    st.info("No product columns found (need product_id/asin/parent_asin and/or product_title).")

# -----------------------------
# Trends over time
# -----------------------------
st.subheader("Sentiment trends over time")
if COL_TIME and COL_LABEL and len(filtered) > 0 and filtered[COL_TIME].notna().any():
    ts = filtered[[COL_TIME, COL_LABEL]].dropna().copy()
    ts["month"] = pd.to_datetime(ts[COL_TIME], errors="coerce").dt.to_period("M").astype(str)
    monthly = ts.groupby(["month", COL_LABEL]).size().rename("count").reset_index()

    if len(monthly) > 0:
        line = alt.Chart(monthly).mark_line(point=True).encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("count:Q"),
            color=alt.Color(f"{COL_LABEL}:N"),
            tooltip=["month:T", f"{COL_LABEL}:N", "count:Q"]
        ).properties(height=280)
        st.altair_chart(line, use_container_width=True)
    else:
        st.caption("No monthly counts after filtering.")
else:
    st.caption("No usable timestamp column for trends.")

# -----------------------------
# Single review prediction (toy)
# -----------------------------
with st.expander("Single review prediction (demo)", expanded=False):
    t = st.text_area("Paste review text", placeholder="E.g., 'Love this lipstick — color pops and lasts!'")
    rating = st.slider("Rating (if known)", 1, 5, 3)
    if st.button("Predict sentiment"):
        text = str(t).lower()
        pos = sum(w in text for w in ["good", "great", "love", "amazing", "smooth", "beautiful", "best"])
        neg = sum(w in text for w in ["bad", "hate", "terrible", "broken", "worst", "sticky", "dry"])
        score = pos - neg + (rating - 3)
        label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
        st.success(f"Predicted (demo): **{label}**")

st.markdown("---")
st.caption(
    "Fast mode uses a demo sample for reliability on Streamlit Cloud. "
    "Switch to **Full** and provide a local path or Drive file ID to analyze the complete dataset."
)
