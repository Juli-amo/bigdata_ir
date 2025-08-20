# Streamlit front-end for the Image Recommender.
# Start with:  streamlit run streamlit_app.py

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Optional OpenCV for decoding previews
try:
    import cv2
except Exception as e:
    st.error(f"Could not import OpenCV: {e}")
    raise

# Local modules
from ImageDatabase import ImageDatabase
from ImageRecommender import ImageRecommender

# ----------------------------- Page setup ------------------------------------
st.set_page_config(page_title="Image Recommender", layout="wide")
st.title("🔎 Image Recommender — Demo UI")


# --------------------------- Helper functions --------------------------------
@st.cache_resource(show_spinner=False)
def load_recommender(db_path: str) -> ImageRecommender:
    """(Legacy) Create & cache the recommender without on-disk ANN cache."""
    db = ImageDatabase(db_path=db_path)
    rec = ImageRecommender(database=db)
    return rec


@st.cache_resource(show_spinner="Building/loading indices …")
def get_recommender(db_path: str = "image_recommender.db", cache_dir: str = ".cache") -> ImageRecommender:
    """Create & cache the recommender with HNSW on-disk cache."""
    db = ImageDatabase(db_path=db_path)
    rec = ImageRecommender(database=db, cache_dir=cache_dir)
    return rec


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative weights to sum to 1. Fallback to deep if all zero."""
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        return {"color": 0.0, "embedding": 1.0, "custom": 0.0}
    return {k: float(max(0.0, v)) / s for k, v in w.items()}


def set_rec_weights(rec: ImageRecommender, w: dict[str, float]) -> None:
    """Apply normalized weights on the recommender."""
    rec.weights = normalize_weights(w)


def bgr_from_upload(file) -> Optional[np.ndarray]:
    """Convert uploaded bytes to BGR image (np.ndarray)."""
    if file is None:
        return None
    data = file.read()
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """BGR → RGB for display."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_rgb_from_path(p: str) -> Optional[np.ndarray]:
    """Load image from disk for result display (RGB)."""
    if not p or not Path(p).exists():
        return None
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr_to_rgb(bgr)


def _try_close_browser_tab() -> None:
    """Try to close the current browser tab gracefully."""
    components.html(
        """
        <script>
        (function () {
          try { window.open('', '_self'); } catch(e) {}
          try { window.close(); } catch(e) {}
          setTimeout(() => {
            try { window.location.replace('about:blank'); } catch(e) {}
          }, 150);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _graceful_exit(delay_s: float = 0.4) -> None:
    """Close resources and terminate the Streamlit process."""
    try:
        rec = st.session_state.get("rec")
        if rec:
            if hasattr(rec, "close") and callable(rec.close):
                rec.close()
            elif getattr(rec, "db", None) and hasattr(rec.db, "close"):
                rec.db.close()
    except Exception as e:
        st.warning(f"Cleanup issue: {e}")
    time.sleep(delay_s)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)


# ------------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    db_path = st.text_input("SQLite database", value="image_recommender.db")
    cache_dir = st.text_input("Index cache directory", value=".cache")

    c1, c2 = st.columns(2)
    with c1:
        reload_btn = st.button("🔄 Load recommender", use_container_width=True)
    with c2:
        show_stats_btn = st.button("📊 DB stats", use_container_width=True)

    with st.expander("Advanced"):
        candidates = st.slider("Candidates (for re-ranking)", 50, 1000, 200, 50)
        topk = st.slider("Top-k results", 1, 12, 5, 1)

    st.divider()
    if st.button("🛑 Exit", use_container_width=True):
        _try_close_browser_tab()
        _graceful_exit(delay_s=0.4)


# -------------------------- Load / reload recommender ------------------------
if "rec" not in st.session_state or reload_btn:
    with st.spinner("Loading recommender & building/loading indices ..."):
        try:
            # Use cached HNSW indices from disk
            st.session_state.rec = get_recommender(db_path=db_path, cache_dir=cache_dir)
        except Exception as e:
            st.error(f"Failed to load recommender: {e}")
            st.stop()

rec: ImageRecommender = st.session_state.rec

# Stats on demand
if show_stats_btn:
    try:
        db_stats = rec.db.get_database_stats()
        sys_stats = rec.get_system_stats()
        st.sidebar.success(
            f"DB: {db_stats['total_images']} imgs | "
            f"Color: {db_stats['color_features_count']} | "
            f"Adv: {db_stats['advanced_features_count']} | "
            f"Deep-Index: {sys_stats['deep_index_size']} | "
            f"Cheap-Index: {sys_stats['cheap_index_size']}"
        )
    except Exception as e:
        st.sidebar.error(f"Stats error: {e}")


# ------------------------- Query configuration (weights) ---------------------
st.subheader("1) Configure query")

preset = st.radio(
    "Which features should contribute to similarity?",
    ["All", "Deep only", "Color only", "Custom/pHash only", "Custom weights"],
    horizontal=True,
)

if preset == "All":
    weights = {"color": 0.4, "embedding": 0.4, "custom": 0.2}
elif preset == "Deep only":
    weights = {"color": 0.0, "embedding": 1.0, "custom": 0.0}
elif preset == "Color only":
    weights = {"color": 1.0, "embedding": 0.0, "custom": 0.0}
elif preset == "Custom/pHash only":
    weights = {"color": 0.0, "embedding": 0.0, "custom": 1.0}
else:
    c = st.slider("Weight: Color", 0.0, 1.0, 0.3, 0.05)
    e = st.slider("Weight: Deep Embedding", 0.0, 1.0, 0.5, 0.05)
    u = st.slider("Weight: Custom/pHash", 0.0, 1.0, 0.2, 0.05)
    weights = {"color": c, "embedding": e, "custom": u}

set_rec_weights(rec, weights)
st.caption(f"Active weights → {rec.weights}")

# Hint if deep embeddings are not yet present
if rec.get_system_stats().get("deep_index_size", 0) == 0:
    st.info(
        "ℹ️ No deep embeddings found in the DB — the cheap index will be used. "
        "Run the backfill first: `python EmbedBackfill.py --db image_recommender.db`."
    )

# ------------------------------ Query mode (tabs) ----------------------------
st.subheader("2) Choose query mode")
tab_single, tab_multi = st.tabs(["Single query", "Multi-query"])

with tab_single:
    uploaded_single = st.file_uploader(
        "Drag & drop or choose a file (jpg/png/…)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        key="u_single",
    )
    go_single = st.button("🚀 Run search (single)", type="primary", use_container_width=True)

    if go_single:
        q_bgr = bgr_from_upload(uploaded_single)
        if q_bgr is None:
            st.error("Could not read the uploaded image.")
            st.stop()

        q_rgb = bgr_to_rgb(q_bgr)

        t0 = time.time()
        try:
            results: list[dict] = rec.find_similar_images(q_bgr, top_k=topk, candidates=candidates)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()
        elapsed = time.time() - t0

        st.subheader("3) Results")
        st.markdown("**Query**")
        st.image(q_rgb, caption=f"Upload • {uploaded_single.name}", use_container_width=True)

        st.markdown(f"**Top-{topk} similar images**  ·  Duration: {elapsed:.3f}s")
        cols = st.columns(topk)
        for i, r in enumerate(results[:topk]):
            meta = r.get("metadata") or {}
            fp = meta.get("filepath", "")
            rgb = load_rgb_from_path(fp)
            cap = f"{meta.get('filename', '?')} • Score: {r.get('similarity_score', 0):.2f}"
            with cols[i]:
                if rgb is not None:
                    st.image(rgb, caption=cap, use_container_width=True)
                else:
                    st.write("⚠️ Could not load result image.")
                    st.caption(cap)

                with st.expander("Details", expanded=False):
                    ds = r.get("detailed_scores") or {}
                    st.json(
                        {
                            "color_similarity": round(ds.get("color_similarity", 0.0), 3),
                            "embedding_similarity": round(ds.get("embedding_similarity", 0.0), 3),
                            "custom_similarity": round(ds.get("custom_similarity", 0.0), 3),
                        }
                    )
                    st.json(
                        {
                            "image_id": r.get("image_id"),
                            "path": fp,
                            "size_bytes": meta.get("file_size"),
                            "shape": (meta.get("height"), meta.get("width"), meta.get("channels")),
                        }
                    )

with tab_multi:
    uploads = st.file_uploader(
        "Drag & drop / choose multiple files",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="u_multi",
    )
    combine = st.selectbox("Combine queries", ["mean", "max"], index=0)
    go_multi = st.button("🚀 Run search (multi)", type="primary", use_container_width=True)

    if go_multi:
        files = uploads or []
        imgs_bgr: list[np.ndarray] = []
        for f in files:
            im = bgr_from_upload(f)
            if im is not None:
                imgs_bgr.append(im)
        if not imgs_bgr:
            st.error("Please upload at least one image.")
            st.stop()

        t0 = time.time()
        try:
            results = rec.find_similar_multi_input(
                imgs_bgr, top_k=topk, candidates=candidates, combine=combine
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()
        elapsed = time.time() - t0

        st.subheader("3) Results")
        st.markdown("**Queries**")
        qcols = st.columns(min(6, len(imgs_bgr)))
        for i, im_bgr in enumerate(imgs_bgr[:6]):
            with qcols[i]:
                st.image(bgr_to_rgb(im_bgr), caption=f"Query {i+1}", use_container_width=True)

        st.markdown(f"**Top-{topk} similar images**  ·  Duration: {elapsed:.3f}s")
        cols = st.columns(topk)
        for i, r in enumerate(results[:topk]):
            meta = r.get("metadata") or {}
            fp = meta.get("filepath", "")
            rgb = load_rgb_from_path(fp)
            cap = f"{meta.get('filename', '?')} • Score: {r.get('similarity_score', 0):.2f}"
            with cols[i]:
                if rgb is not None:
                    st.image(rgb, caption=cap, use_container_width=True)
                else:
                    st.write("⚠️ Could not load result image.")
                    st.caption(cap)

                with st.expander("Details", expanded=False):
                    ds = r.get("detailed_scores") or {}
                    st.json(
                        {
                            "color_similarity": round(ds.get("color_similarity", 0.0), 3),
                            "embedding_similarity": round(ds.get("embedding_similarity", 0.0), 3),
                            "custom_similarity": round(ds.get("custom_similarity", 0.0), 3),
                        }
                    )
                    st.json(
                        {
                            "image_id": r.get("image_id"),
                            "path": fp,
                            "size_bytes": meta.get("file_size"),
                            "shape": (meta.get("height"), meta.get("width"), meta.get("channels")),
                        }
                    )

# ------------------------------ Footer ---------------------------------------
st.write("---")
st.caption(
    "Tip: For best quality, make sure the database already contains deep embeddings "
    "(`python EmbedBackfill.py --db image_recommender.db`). You can adjust feature weights above."
)