import numpy as np

from FeatureExtraction import ColorAnalyzer, compute_phash


def test_color_features_shape():
    img = np.full((100, 150, 3), 200, dtype=np.uint8)
    ca = ColorAnalyzer(k_clusters=3)
    cf = ca.extract_color_features(img, num_bins=32, num_dominant=3)
    assert len(cf["dominant_colors"]) == 3
    assert len(cf["hsv_histogram"]) == 3
    assert len(cf["bgr_histogram"]) == 3


def test_phash_deterministic():
    img = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    h1 = compute_phash(img)
    h2 = compute_phash(img.copy())
    assert h1 == h2
