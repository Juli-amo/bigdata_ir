# tests/test_similarity.py
import numpy as np
from ImageRecommender import SimilarityCalculator

def _flat_hist(n=16):
    h = (np.ones(n)/n).tolist()
    return [h,h,h]

def _cf(bright=128.0, std=[10,10,10]):
    return {
        "hsv_histogram": _flat_hist(16),
        "bgr_histogram": _flat_hist(16),
        "color_stats": {"brightness": float(bright), "mean_bgr": [120,130,140], "std_bgr": std},
        "dominant_colors": [[200,50,50],[50,200,50],[50,50,200]],
    }

def test_color_similarity_identical():
    sc = SimilarityCalculator()
    f = _cf()
    assert sc.color_sim(f,f) > 0.99

def test_color_similarity_brightness_penalty():
    sc = SimilarityCalculator()
    f1, f2 = _cf(bright=20.0), _cf(bright=240.0)
    assert sc.color_sim(f1,f2) < 0.8  # Helligkeitsabzug sichtbar

def test_emb_sim_deep_present():
    sc = SimilarityCalculator()
    a = np.ones(128, np.float32); b = np.ones(128, np.float32)
    assert sc.emb_sim(a,b,{}, {}) > 0.99

def test_emb_sim_fallback_texture():
    sc = SimilarityCalculator()
    t1 = {"texture_features":{"mean":0.5,"std":0.1,"variance":0.01}}
    t2 = {"texture_features":{"mean":0.5,"std":0.1,"variance":0.01}}
    # keine Deep-Embeddings -> Texture-Fallback
    assert sc.emb_sim(None, None, t1, t2) > 0.9

def test_emb_sim_fallback_no_texture():
    sc = SimilarityCalculator()
    # Weder Deep noch Texture -> definierter 0.5-Return im Code
    assert abs(sc.emb_sim(None, None, {}, {} ) - 0.5) < 1e-6

def test_custom_sim_extremes():
    sc = SimilarityCalculator()
    cf = _cf()
    adv = {"texture_features":{"mean":0.5}}
    # phash maximal gleich vs. maximal verschieden
    sim_eq = sc.custom_sim(adv, adv, cf, cf, "0x0", "0x0")
    sim_ne = sc.custom_sim(adv, adv, cf, cf, "0x0", "0xffffffffffffffff")
    assert sim_eq > sim_ne