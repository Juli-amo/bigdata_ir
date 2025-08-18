# Unit tests for SimilarityCalculator

from __future__ import annotations

import numpy as np
import cv2
import pytest

from FeatureExtraction import ColorAnalyzer, ImageFeatureExtractor, compute_phash
from ImageRecommender import SimilarityCalculator


def _flat_hist(n=16):
    return (np.ones(n) / n).tolist()

def _solid(bgr: tuple[int, int, int], w: int = 64, h: int = 64) -> np.ndarray:
    """Create a solid-color BGR image (uint8)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = np.array(bgr, dtype=np.uint8)
    return img

def _cf(bright: float = 128.0, bins: int = 16) -> dict:
    """Build a synthetic color feature dict with uniform histograms and given brightness."""
    hist = (np.ones(bins) / bins).tolist()
    return {
        "hsv_histogram": [hist, hist, hist],
        "bgr_histogram": [hist, hist, hist],
        "color_stats": {
            "brightness": float(bright),
            "mean_bgr": [120.0, 130.0, 140.0],
            "std_bgr": [10.0, 10.0, 10.0],
        },
    }



def test_color_similarity_identical():
    sc = SimilarityCalculator()
    f = _cf(128.0)
    assert sc.color_sim(f, f) > 0.99


def test_color_similarity_brightness_penalty():
    # With identical histograms but large brightness gap, the small brightness term (10%)
    # should reduce the score slightly. With your weighting it is ~0.914.
    sc = SimilarityCalculator()
    f1 = _cf(bright=20.0)
    f2 = _cf(bright=240.0)
    sim = sc.color_sim(f1, f2)
    assert sim == pytest.approx(0.914, abs=0.02)  # visible penalty, but still high


def test_embedding_similarity_equal_vectors():
    sc = SimilarityCalculator()
    a = np.ones(128, dtype=np.float32)
    b = np.ones(128, dtype=np.float32)
    assert sc.emb_sim(a, b, {}, {}) > 0.99


def test_embedding_similarity_fallback_texture():
    sc = SimilarityCalculator()
    # no deep embeddings -> fallback to texture mean/std/var cosine in [0,1]
    f1 = {"texture_features": {"mean": 10.0, "std": 2.0, "variance": 4.0}}
    f2 = {"texture_features": {"mean": 10.0, "std": 2.0, "variance": 4.0}}
    assert sc.emb_sim(None, None, f1, f2) > 0.99


def test_custom_similarity_phash_effect():
    sc = SimilarityCalculator()
    # Same color stats/hists → pHash dominates the difference
    fC = _cf(128.0)
    # force different phash hex strings
    sim_equal = sc.custom_sim({}, {}, fC, fC, "0xaaaaaaaaaaaaaaaa", "0xaaaaaaaaaaaaaaaa")
    sim_diff = sc.custom_sim({}, {}, fC, fC, "0xaaaaaaaaaaaaaaaa", "0x5555555555555555")
    assert sim_equal > sim_diff

def test_texture_features_basic_stats():
    """Texture extractor returns stable keys with finite numbers."""
    fe = ImageFeatureExtractor()
    img = _solid((120, 40, 200))
    out = fe.extract_texture_features(img)
    for k in ("mean", "std", "variance", "texture_hist", "lap_var", "entropy"):
        assert k in out
    assert len(out["texture_hist"]) == 32
    # Values should be finite
    scalars = [out["mean"], out["std"], out["variance"], out["lap_var"], out["entropy"]]
    assert all(np.isfinite(s) for s in scalars)


def test_phash_equal_and_robust():
    """pHash should be identical for identical images and change with perturbation."""
    img = _solid((0, 0, 255))
    h1 = compute_phash(img)
    h2 = compute_phash(img.copy())
    assert isinstance(h1, int) and isinstance(h2, int)
    assert h1 == h2

    # Small change → often flips some bits
    noisy = img.copy()
    cv2.circle(noisy, (8, 8), 4, (0, 0, 200), -1)
    h3 = compute_phash(noisy)
    assert h3 != h1