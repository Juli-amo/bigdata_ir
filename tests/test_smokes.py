# tests/test_smokes.py
# Smoke tests: quick end-to-end checks that don’t touch the big dataset.
# Run: pytest -q -m smoke

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

# Mark all tests here as "smoke" (to filter via `-m smoke`).
# Optional: add a pytest.ini to register this marker to avoid warnings.
pytestmark = pytest.mark.smoke


def _maybe_import_cv2():
    """Import cv2 or skip the test if OpenCV is not installed."""
    try:
        import cv2  # type: ignore
    except Exception:
        pytest.skip("OpenCV (cv2) not available")
    return cv2


def _make_tiny_jpeg(path: Path, bgr=(0, 0, 255)) -> None:
    """Create a tiny 16x16 JPEG (BGR) for quick I/O tests."""
    cv2 = _maybe_import_cv2()
    img = np.full((16, 16, 3), bgr, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, "Failed to write tiny JPEG"


def _bootstrap_minidb(db_path: Path, images: list[tuple[str, Path]]) -> None:
    """Create schema via ImageDatabase and insert minimal rows for provided images."""
    from ImageDatabase import ImageDatabase  # noqa: WPS433

    db = ImageDatabase(str(db_path))  # creates tables + pragmas
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    for image_id, file_path in images:
        cur.execute(
            """
            INSERT INTO images (image_id, filename, filepath, file_size, width, height, channels, file_hash, photographer, created_date, added_to_db)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, datetime('now'), datetime('now'))
            """,
            (
                image_id,
                file_path.name,
                str(file_path),
                file_path.stat().st_size,
                16,
                16,
                3,
            ),
        )
        # Minimal color features (flat histograms)
        cur.execute(
            """
            INSERT INTO color_features (image_id, dominant_colors, hsv_histogram, bgr_histogram, color_stats)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                image_id,
                json.dumps([[0, 0, 0]] * 3),
                json.dumps([[0.0] * 16, [0.0] * 16, [0.0] * 16]),
                json.dumps([[0.0] * 16, [0.0] * 16, [0.0] * 16]),
                json.dumps(
                    {
                        "brightness": 128.0,
                        "mean_bgr": [0, 0, 0],
                        "std_bgr": [0, 0, 0],
                    }
                ),
            ),
        )
        # Tiny random 128D embedding as blob
        z = np.random.RandomState(0).randn(128).astype(np.float32)
        cur.execute(
            """
            INSERT INTO advanced_features (image_id, texture_features, deep_embeddings, custom_features, deep_dim)
            VALUES (?, NULL, ?, NULL, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                deep_embeddings=excluded.deep_embeddings,
                deep_dim=excluded.deep_dim
            """,
            (image_id, memoryview(z.tobytes()), 128),
        )

    con.commit()
    con.close()

    stats = db.get_database_stats()
    assert stats["total_images"] >= len(images)


def _extract_json_stdout(stdout: str):
    """Extract JSON payload from stdout that may contain log lines.

    Strategy:
    1) Try strict parse of the whole stdout.
    2) Otherwise, scan lines and skip known log prefixes like [INFO], [WARN], etc.
       Parse from the first line whose first non-space char is '[' or '{' AND
       which is not a log prefix.
    3) As a fallback, try to locate the first plausible JSON start with a regex.
    """
    import re

    text = stdout.strip()
    # 1) Try strict parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Line-wise scan, skip log prefixes
    LOG_PREFIXES = ("[INFO", "[WARN", "[WARNING", "[ERROR", "[DEBUG", "[TRACE]")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        ls = line.lstrip()
        if not ls:
            continue
        if ls.startswith(LOG_PREFIXES):
            continue
        if ls[0] in "[{]":
            payload = "\n".join(lines[i:])
            try:
                return json.loads(payload)
            except Exception:
                # keep searching; there might be more logs mixed in
                continue

    # 3) Fallback regex: find first '[' or '{' that is followed soon by '{' or '['
    m = re.search(r"(\[|\{)\s*(\{|\[)", text)
    if m:
        try:
            return json.loads(text[m.start():])
        except Exception:
            pass

    raise AssertionError(f"No JSON payload found in stdout:\n{stdout}")


def test_cli_stats_smoke(tmp_path: Path) -> None:
    """`cli.py stats` should print valid JSON even on a fresh DB."""
    db_path = tmp_path / "mini.db"

    from ImageDatabase import ImageDatabase  # noqa: WPS433

    ImageDatabase(str(db_path))

    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        env.get("PYTHONPATH", "")
        + (os.pathsep if "PYTHONPATH" in env else "")
        + str(Path.cwd())
    )

    proc = subprocess.run(
        [sys.executable, "cli.py", "stats", "--db", str(db_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    data = _extract_json_stdout(proc.stdout)
    assert "total_images" in data


def test_cli_query_smoke_minidb(tmp_path: Path) -> None:
    """`cli.py query` returns a JSON list with minimal DB and a tiny query image."""
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _make_tiny_jpeg(img_a, bgr=(0, 0, 255))
    _make_tiny_jpeg(img_b, bgr=(0, 255, 0))

    db_path = tmp_path / "mini.db"
    _bootstrap_minidb(db_path, images=[("ida", img_a), ("idb", img_b)])

    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        env.get("PYTHONPATH", "")
        + (os.pathsep if "PYTHONPATH" in env else "")
        + str(Path.cwd())
    )

    proc = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "query",
            "--image",
            str(img_a),
            "--db",
            str(db_path),
            "--topk",
            "2",
            "--candidates",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr

    results = _extract_json_stdout(proc.stdout)
    assert isinstance(results, list)
    if results:
        r0 = results[0]
        assert "similarity_score" in r0
        assert 0.0 <= float(r0["similarity_score"]) <= 1.0
        meta = r0.get("metadata") or {}
        assert "filename" in meta