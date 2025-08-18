# tests/conftest.py
# Shared fixtures for all tests: ensure import path, create a tiny temp DB with images,
# and stub the VisionEmbedder for fast, deterministic tests.

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


# --- make project root importable -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="function")
def tmp_db_with_images(tmp_path):
    """
    Create 8 small synthetic JPEGs, ingest them into a fresh SQLite DB via ImageDatabase,
    and return (db_path, [image_paths]).
    """
    from ImageDatabase import ImageDatabase

    img_dir = tmp_path / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(0)
    paths = []
    for i in range(8):
        # colored base + tiny noise
        base = np.zeros((120, 160, 3), np.uint8)
        base[:] = (i * 30 % 255, (i * 60) % 255, (i * 90) % 255)
        noise = (rng.randn(*base.shape) * 3).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        p = img_dir / f"im_{i:02d}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        paths.append(p)

    db_path = tmp_path / "test.db"
    db = ImageDatabase(str(db_path))
    stats = db.bulk_add_directory(
        str(img_dir),
        recursive=False,
        workers=4,
        pool="thread",
        decode_mode="opencv",
        jpeg_reduce=8,
        log_every=0,
    )
    # at least most images should be added
    assert stats["added"] >= 6
    return str(db_path), [str(p) for p in paths]


@pytest.fixture
def stub_embedder(monkeypatch):
    """
    Replace DeepEmbedding.VisionEmbedder with a lightweight stub that returns
    deterministic 128D vectors quickly. Keeps tests fast and reproducible.
    """
    import DeepEmbedding

    class _Stub:
        def __init__(self, *a, **k):
            self.device = "cpu"

        def available(self) -> bool:
            return True

        def embed_array(self, img):
            # simple constant vector; length matches proj_dim=128
            return np.ones(128, dtype=np.float32)

        def embed_batch(self, imgs):
            return [np.ones(128, dtype=np.float32) for _ in imgs]

    monkeypatch.setattr(DeepEmbedding, "VisionEmbedder", _Stub)