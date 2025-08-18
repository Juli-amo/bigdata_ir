# FeatureExtraction.py
# Color & texture features + perceptual hash (pHash).
# All functions have short, English docstrings and return plain Python/NumPy types.

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

__all__ = [
    "ColorAnalyzer",
    "ImageFeatureExtractor",
    "compute_phash",
]


class ColorAnalyzer:
    """Compute robust color features (dominant colors, histograms, stats).

    - Works on BGR images (uint8) as returned by OpenCV.
    - Shrinks images to `max_side` for speed (quality impact is negligible).
    - Dominant colors are computed via k-means in Lab space for stability.
    """

    def __init__(self, k_clusters: int = 3, random_state: int = 42, max_side: int = 256) -> None:
        self.k_clusters = int(k_clusters)
        self.random_state = int(random_state)
        self.max_side = int(max_side)

    # ---------------------- internal helpers ----------------------

    def _ensure_bgr3(self, img: np.ndarray) -> np.ndarray:
        """Ensure 3-channel BGR (convert gray/BGRA if needed)."""
        if img is None:
            raise ValueError("image is None")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _shrink(self, img: np.ndarray) -> np.ndarray:
        """Resize long side to `max_side` (INTER_AREA) if the image is large."""
        h, w = img.shape[:2]
        s = max(h, w)
        if s <= self.max_side:
            return img
        scale = self.max_side / float(s)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _calc_norm_hist(
        self, image: np.ndarray, channel: int, num_bins: int, range_max: int
    ) -> np.ndarray:
        """Normalized 1D histogram for a single channel."""
        hist = cv2.calcHist([image], [channel], None, [int(num_bins)], [0, int(range_max)]).astype(
            np.float32
        )
        return (hist / (float(hist.sum()) + 1e-8)).flatten()

    # ------------------------ public API -------------------------

    def color_histogram_similarity(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        color_space: str = "HSV",
        num_bins: int = 32,
        use_opencv: bool = False,
        show_plots: bool = False,
    ) -> float:
        """Cosine similarity of per-channel histograms (HSV or BGR) in [0,1].

        Args:
            image1, image2: input BGR images.
            color_space: "HSV" (default) or "BGR".
            num_bins: bins per channel.
            use_opencv: use cv2.compareHist correlation (mapped to [0,1]).
            show_plots: optional visualization (matplotlib; lazy import).

        Returns:
            Mean similarity across channels in [0,1].
        """
        img1 = self._shrink(self._ensure_bgr3(image1))
        img2 = self._shrink(self._ensure_bgr3(image2))

        if color_space.upper() == "HSV":
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            ranges = [180, 256, 256]  # H, S, V
            channels = ["H", "S", "V"]
        else:
            ranges = [256, 256, 256]
            channels = ["B", "G", "R"]

        if show_plots:
            import matplotlib.pyplot as plt  # lazy import

        sims: list[float] = []
        for i, ch in enumerate(channels):
            h1 = self._calc_norm_hist(img1, i, num_bins, ranges[i])
            h2 = self._calc_norm_hist(img2, i, num_bins, ranges[i])

            if use_opencv:
                # cv2.HISTCMP_CORREL returns [-1,1] → map to [0,1]
                raw = float(
                    cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_CORREL)
                )
                sim = max(0.0, min(1.0, (raw + 1.0) * 0.5))
            else:
                num = float(np.dot(h1, h2))
                den = float(np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
                sim = max(0.0, min(1.0, num / den))
            sims.append(sim)

            if show_plots:
                plt.figure(figsize=(6, 3))
                plt.plot(h1, label=f"{ch} image1", alpha=0.7)
                plt.plot(h2, label=f"{ch} image2", alpha=0.7)
                plt.title(f"Histogram channel {ch}")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()

        return float(np.mean(sims))

    def get_dominant_colors(self, image: np.ndarray, max_samples: int = 10_000) -> np.ndarray:
        """Dominant colors via k-means in Lab (returns BGR centers, shape [K,3])."""
        bgr = self._shrink(self._ensure_bgr3(image))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype("float32")
        n = lab.shape[0]
        if n == 0:
            return np.zeros((self.k_clusters, 3), dtype=np.float32)

        # Subsample pixels for speed on large images
        if n > max_samples:
            rs = np.random.RandomState(self.random_state)
            idx = rs.choice(n, size=max_samples, replace=False)
            lab_sample = lab[idx]
        else:
            lab_sample = lab

        K = max(1, min(self.k_clusters, lab_sample.shape[0]))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _compact, _labels, centers_lab = cv2.kmeans(
            lab_sample, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        centers_lab_u8 = np.clip(centers_lab, 0, 255).astype("uint8").reshape(-1, 1, 3)
        centers_bgr = (
            cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2BGR).reshape(K, 3).astype("float32")
        )

        # Sort by brightness to get a stable ordering
        order = np.argsort(centers_bgr.sum(axis=1))
        return centers_bgr[order]

    def calculate_color_distance(self, colors1: np.ndarray, colors2: np.ndarray) -> float:
        """Euclidean distance between two color center sets."""
        return float(np.linalg.norm(colors1 - colors2))

    def extract_color_features(
        self, image: np.ndarray, num_bins: int = 32, num_dominant: int = 3
    ) -> dict[str, Any]:
        """Aggregate color features (histograms, stats, dominant colors)."""
        bgr = self._shrink(self._ensure_bgr3(image))

        # BGR histograms
        b = cv2.calcHist([bgr], [0], None, [num_bins], [0, 256]).astype("float32")
        g = cv2.calcHist([bgr], [1], None, [num_bins], [0, 256]).astype("float32")
        r = cv2.calcHist([bgr], [2], None, [num_bins], [0, 256]).astype("float32")

        def _norm(h: np.ndarray) -> list[float]:
            return (h / (float(h.sum()) + 1e-8)).flatten().tolist()

        bgr_hist = [_norm(b), _norm(g), _norm(r)]

        # HSV histograms
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [num_bins], [0, 180]).astype("float32")
        s = cv2.calcHist([hsv], [1], None, [num_bins], [0, 256]).astype("float32")
        v = cv2.calcHist([hsv], [2], None, [num_bins], [0, 256]).astype("float32")
        hsv_hist = [_norm(h), _norm(s), _norm(v)]

        # Basic stats
        brightness = float(hsv[:, :, 2].mean())
        mean_bgr = [float(bgr[:, :, c].mean()) for c in range(3)]
        std_bgr = [float(bgr[:, :, c].std()) for c in range(3)]
        stats = {"brightness": brightness, "mean_bgr": mean_bgr, "std_bgr": std_bgr}

        # Dominant colors (k-means in Lab)
        centers = self.get_dominant_colors(bgr)
        if centers.shape[0] < num_dominant:
            pad = np.tile(
                centers.mean(axis=0, keepdims=True), (num_dominant - centers.shape[0], 1)
            )
            centers = np.vstack([centers, pad])
        centers = centers[: int(num_dominant)]

        return {
            "dominant_colors": centers.astype("float32").tolist(),
            "hsv_histogram": hsv_hist,
            "bgr_histogram": bgr_hist,
            "color_stats": stats,
        }


class ImageFeatureExtractor:
    """Simple texture features based on gradients, Laplacian and entropy."""

    def extract_texture_features(self, image: np.ndarray, num_bins: int = 32) -> dict[str, Any]:
        """Return basic texture statistics and gradient histogram."""
        if image is None:
            raise ValueError("image is None")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Shrink for speed
        h, w = gray.shape[:2]
        s = max(h, w)
        if s > 256:
            scale = 256.0 / float(s)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Gradients (Sobel) and magnitude histogram
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        t_hist = cv2.calcHist([mag], [0], None, [num_bins], [0, 255]).astype("float32")
        t_hist = (t_hist / (t_hist.sum() + 1e-8)).flatten().tolist()

        # Laplacian focus measure
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # Entropy of gray histogram
        hgray = cv2.calcHist([gray], [0], None, [256], [0, 256]).astype("float32").flatten()
        p = hgray / (hgray.sum() + 1e-8)
        mask = p > 0
        entropy = float(-(p[mask] * np.log2(p[mask])).sum())

        return {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "variance": float(np.var(gray)),
            "texture_hist": t_hist,
            "lap_var": lap_var,
            "entropy": entropy,
        }


def compute_phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Perceptual hash (64-bit default) using DCT low frequencies.

    Args:
        image: BGR or grayscale image.
        hash_size: number of low-frequency DCT coeffs per dimension.
        highfreq_factor: resize factor before DCT to capture more frequencies.

    Returns:
        Integer bitstring (e.g., 64-bit when hash_size=8).
    """
    if image is None:
        raise ValueError("image is None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    size = int(hash_size) * int(highfreq_factor)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR).astype("float32")
    dct = cv2.dct(gray)
    low = dct[:hash_size, :hash_size]
    med = float(np.median(low))
    bits = (low > med).astype(np.uint8).flatten()

    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)
