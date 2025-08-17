# Streamlit front-end for the Image Recommender.
# Start with:  streamlit run streamlit_app.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, List

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
    """Create & cache the recommender (builds indices once)."""
    db = ImageDatabase(db_path=db_path)
    rec = ImageRecommender(database=db)  # ANN threshold handled inside
    return rec


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Normalize non-negative weights to sum to 1. Fallback to deep if all zero."""
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        return {"color": 0.0, "embedding": 1.0, "custom": 0.0}
    return {k: float(max(0.0, v)) / s for k, v in w.items()}


def set_rec_weights(rec: ImageRecommender, w: Dict[str, float]) -> None:
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
        import os, signal

        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        import os

        os._exit(0)


# ------------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    db_path = st.text_input("SQLite database", value="image_recommender.db")
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
    with st.spinner("Loading recommender & building indices ..."):
        try:
            st.session_state.rec = load_recommender(db_path)
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


# ------------------------------ Upload & search ------------------------------
st.subheader("2) Upload image")

uploaded = st.file_uploader(
    "Drag & drop or choose a file (jpg/png/…)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
)

go = st.button("🚀 Run search", type="primary", use_container_width=True, disabled=(uploaded is None))

if go:
    q_bgr = bgr_from_upload(uploaded)
    if q_bgr is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    q_rgb = bgr_to_rgb(q_bgr)

    # Execute query (measure time)
    t0 = time.time()
    try:
        results: List[Dict] = rec.find_similar_images(q_bgr, top_k=topk, candidates=candidates)
    except Exception as e:
        st.error(f"Search failed: {e}")
        st.stop()
    elapsed = time.time() - t0

    # --------------------------- Results UI ----------------------------------
    st.subheader("3) Results")

    # Query preview
    st.markdown("**Query**")
    st.image(q_rgb, caption=f"Upload • {uploaded.name}", width=360)

    # Top-k cards
    st.markdown(f"**Top-{topk} similar images**  ·  Duration: {elapsed:.3f}s")
    cols = st.columns(topk)
    for i, r in enumerate(results[:topk]):
        meta = r.get("metadata") or {}
        fp = meta.get("filepath", "")
        rgb = load_rgb_from_path(fp)
        cap = f"{meta.get('filename','?')} • Score: {r.get('similarity_score',0):.2f}"
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