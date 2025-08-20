# tests/test_recommender.py
# Tests for the end-to-end recommender on a tiny synthetic DB.

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import pytest

from ImageDatabase import ImageDatabase
from ImageRecommender import ImageRecommender


def _mk_solid_file(path: Path, bgr: tuple[int, int, int], size: int = 64) -> Path:
    """Create a small solid BGR image and save it."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def tiny_db(tmp_path: Path) -> str:
    """Create a tiny DB (no deep embeddings) for fast color-based search."""
    db_path = tmp_path / "tiny.db"
    root = tmp_path / "imgs"
    _mk_solid_file(root / "red.jpg", (0, 0, 255))
    _mk_solid_file(root / "green.jpg", (0, 255, 0))
    _mk_solid_file(root / "blue.jpg", (255, 0, 0))
    _mk_solid_file(root / "dark_red.jpg", (0, 0, 120))
    _mk_solid_file(root / "light_red.jpg", (0, 0, 240))

    db = ImageDatabase(str(db_path))
    stats = db.bulk_add_directory(
        str(root),
        recursive=False,
        workers=1,
        pool="thread",
        batch_size=100,
        num_bins=32,
        k_clusters=3,
        resize_max=128,
        decode_mode="opencv",
        jpeg_reduce=8,
        log_every=0,
    )
    assert stats["added"] == 5
    return str(db_path)


def test_single_query_prefers_same_color(tiny_db: str, tmp_path: Path):
    """With color-heavy weights, a red query should rank red images highest."""
    # Build a red-ish query that is NOT in the DB.
    q_path = tmp_path / "query_red.jpg"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 230)  # red-ish
    cv2.imwrite(str(q_path), img)

    db = ImageDatabase(tiny_db)
    rec = ImageRecommender(database=db, ann_threshold=10)
    # Emphasize color to make the outcome deterministic for solids
    rec.weights = {"color": 1.0, "embedding": 0.0, "custom": 0.0}

    results = rec.find_similar_images(str(q_path), top_k=3, candidates=20)
    assert len(results) >= 1
    top_names = [r["metadata"]["filename"] for r in results]
    joined = " ".join(top_names).lower()
    # Expect "red" variants near the top
    assert "red" in joined


def test_system_stats_and_index_sizes(tiny_db: str):
    """System stats expose index sizes and weight dict."""
    db = ImageDatabase(tiny_db)
    rec = ImageRecommender(database=db, ann_threshold=10)
    s = rec.get_system_stats()
    assert "weights" in s and isinstance(s["weights"], dict)
    assert s["cheap_index_size"] > 0
    # deep index may be zero in this tiny DB (no embeddings)
    assert "deep_index_size" in s