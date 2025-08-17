#!/usr/bin/env python3
"""
EmbedBackfill.py

Compute and persist deep embeddings for all images that are missing them.

- Streams file paths from SQLite (WAL, tuned PRAGMAs).
- Decodes images in parallel (TurboJPEG if available, falls back to OpenCV).
- Runs batched forward passes via VisionEmbedder (MPS/GPU/CPU auto).
- Saves 128D float32 vectors as BLOBs (with deep_dim) into advanced_features.
- Can run once (--once) or poll forever (--poll-interval).

Usage (single pass):
  python EmbedBackfill.py --db image_recommender.db --batch 192 --io-workers 16 \
      --decode-mode auto --jpeg-reduce 8 --log-every 5 --once

Usage (polling):
  python EmbedBackfill.py --db image_recommender.db --batch 192 --io-workers 16 \
      --decode-mode auto --jpeg-reduce 8 --log-every 5 --poll-interval 300
"""

from __future__ import annotations

import os
import time
import json
import logging
import sqlite3
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from DeepEmbedding import VisionEmbedder  # uses MPS if available

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

log = logging.getLogger("EmbedBackfill")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# --------------------------------------------------------------------------- #
# Optional TurboJPEG
# --------------------------------------------------------------------------- #

JPEG_EXTS = {".jpg", ".jpeg"}

HAVE_TURBO = False
try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT  # type: ignore

    HAVE_TURBO = True
except Exception:
    HAVE_TURBO = False

_TLS_TURBO: Optional[TurboJPEG] = None


def _get_turbo() -> Optional["TurboJPEG"]:
    """Lazy-initialize TurboJPEG (if installed)."""
    global _TLS_TURBO
    if not HAVE_TURBO:
        return None
    if _TLS_TURBO is None:
        # Let the wrapper find the system lib (or honor $TURBOJPEG)
        _TLS_TURBO = TurboJPEG()
    return _TLS_TURBO


# --------------------------------------------------------------------------- #
# SQLite helpers
# --------------------------------------------------------------------------- #

def _ensure_pragmas(conn: sqlite3.Connection) -> None:
    """Speed-friendly PRAGMAs for WAL SQLite workloads."""
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA temp_store=MEMORY;")
    c.execute("PRAGMA mmap_size=3000000000;")
    c.execute("PRAGMA cache_size=-200000;")
    c.execute("PRAGMA busy_timeout=60000;")
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Ensure advanced_features has deep_dim (idempotent)."""
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE advanced_features ADD COLUMN deep_dim INTEGER")
        conn.commit()
    except Exception:
        pass


def _fetch_missing(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Return (image_id, filepath) for rows without deep_embeddings."""
    c = conn.cursor()
    q = """
      SELECT i.image_id, i.filepath
      FROM images i
      LEFT JOIN advanced_features a ON a.image_id = i.image_id
      WHERE a.deep_embeddings IS NULL OR a.deep_embeddings = X''
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    c.execute(q)
    return [(r[0], r[1]) for r in c.fetchall()]


def _save_batch(conn: sqlite3.Connection, rows: List[Tuple[str, bytes, int]]) -> None:
    """Upsert (image_id, deep_embeddings, deep_dim) into advanced_features."""
    c = conn.cursor()
    c.executemany(
        """
        INSERT INTO advanced_features (image_id, texture_features, deep_embeddings, custom_features, deep_dim)
        VALUES (?, NULL, ?, NULL, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            deep_embeddings = excluded.deep_embeddings,
            deep_dim        = excluded.deep_dim
        """,
        rows,
    )


def _save_batch_robust(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, bytes, int]],
    retries: int = 40,
    sleep_s: float = 0.25,
) -> bool:
    """Handle 'database is locked' with bounded retries."""
    for _ in range(int(retries)):
        try:
            _save_batch(conn, rows)
            return True
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(sleep_s)
                continue
            raise
    return False


# --------------------------------------------------------------------------- #
# Decoders
# --------------------------------------------------------------------------- #

def _read_bytes(path: str) -> Optional[bytes]:
    """Read file into bytes (no buffering for speed)."""
    try:
        with open(path, "rb", buffering=0) as f:
            return f.read()
    except Exception:
        return None


def _decode_turbo(path: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """TurboJPEG decode with compressed-domain scaling (1/1, 1/2, 1/4, 1/8)."""
    tj = _get_turbo()
    if tj is None:
        return None
    data = _read_bytes(path)
    if data is None:
        return None
    sf = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}.get(int(jpeg_reduce), (1, 8))
    try:
        return tj.decode(data, pixel_format=TJPF_BGR, scaling_factor=sf,
                         flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
    except Exception:
        return None


def _decode_imdecode(path: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """cv2.imdecode on bytes; optional AREA downscale for JPEG to emulate reduce."""
    data = _read_bytes(path)
    if data is None:
        return None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and os.path.splitext(path)[1].lower() in JPEG_EXTS and jpeg_reduce in (2, 4, 8):
            f = 1.0 / float(jpeg_reduce)
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def _decode_opencv(path: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """OpenCV imread; for JPEG use reduced decode flags when available."""
    try:
        flag = cv2.IMREAD_COLOR
        ext = os.path.splitext(path)[1].lower()
        if ext in JPEG_EXTS and jpeg_reduce in (2, 4, 8):
            flag = getattr(cv2, f"IMREAD_REDUCED_COLOR_{jpeg_reduce}")
        return cv2.imread(path, flag)
    except Exception:
        return None


def _decode_one(path: str, decode_mode: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """Decode a single image file according to the selected mode."""
    name = os.path.basename(path)
    if name.startswith("._") or name == ".DS_Store":
        return None

    ext = os.path.splitext(path)[1].lower()

    if decode_mode == "turbo" and ext in JPEG_EXTS:
        return _decode_turbo(path, jpeg_reduce)
    if decode_mode == "imdecode":
        return _decode_imdecode(path, jpeg_reduce)
    if decode_mode == "opencv":
        return _decode_opencv(path, jpeg_reduce)

    # auto: try turbo (JPEG) -> imdecode -> opencv
    if ext in JPEG_EXTS and HAVE_TURBO:
        img = _decode_turbo(path, jpeg_reduce)
        if img is not None:
            return img
    img = _decode_imdecode(path, jpeg_reduce)
    return img if img is not None else _decode_opencv(path, jpeg_reduce)


# --------------------------------------------------------------------------- #
# Backfill core
# --------------------------------------------------------------------------- #

def backfill_once(
    db_path: str,
    batch: int = 192,
    io_workers: int = 12,
    decode_mode: str = "auto",
    jpeg_reduce: int = 8,
    log_every: int = 5,
    fetch_limit: int = 5000,
) -> Dict[str, float]:
    """Process up to `fetch_limit` missing items exactly once.

    Returns:
        dict with 'processed', 'saved', 'elapsed_s', 'throughput_imgs_per_s'.
    """
    t_round = time.time()
    conn = sqlite3.connect(db_path)
    _ensure_pragmas(conn)
    _ensure_columns(conn)

    emb = VisionEmbedder(model_name="mobilenet_v3_large", proj_dim=128, seed=42)
    if not emb.available():
        raise RuntimeError("VisionEmbedder not available or device not ready")

    todo = _fetch_missing(conn, limit=fetch_limit)
    if not todo:
        return {"processed": 0, "saved": 0, "elapsed_s": 0.0, "throughput_imgs_per_s": 0.0}

    N = len(todo)
    log.info(f"{N} pending embeddings – starting (device={emb.device}, batch={batch}, io_workers={io_workers})")

    saved = 0
    processed = 0
    last_log = time.time()
    start = time.time()

    # Decode in chunks to control memory footprint
    chunk_size = io_workers * batch
    for ci in range(0, N, chunk_size):
        chunk = todo[ci : ci + chunk_size]

        # Parallel decoding
        imgs: List[Optional[np.ndarray]] = [None] * len(chunk)
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            futures = {ex.submit(_decode_one, p, decode_mode, jpeg_reduce): j for j, (_, p) in enumerate(chunk)}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    imgs[j] = fut.result()
                except Exception:
                    imgs[j] = None

        # Embed & save in micro-batches
        for bi in range(0, len(chunk), batch):
            sub = chunk[bi : bi + batch]
            sub_imgs = imgs[bi : bi + batch]

            valid = [(k, im) for k, im in enumerate(sub_imgs) if im is not None]
            processed += len(sub_imgs)
            if not valid:
                # nothing decodable in this micro-batch
                # periodic logging still applies below
                pass
            else:
                idx_local, imgs_valid = zip(*valid)
                try:
                    Z = emb.embed_batch(list(imgs_valid))
                except Exception as e:
                    log.warning(f"embed_batch failed (skip micro-batch): {e}")
                    Z = None

                if Z is not None and len(Z) > 0:
                    rows: List[Tuple[str, bytes, int]] = []
                    for off, zi in zip(idx_local, Z):
                        iid = sub[off][0]
                        rows.append((iid, memoryview(zi.tobytes()), int(zi.shape[0])))

                    if rows and _save_batch_robust(conn, rows):
                        conn.commit()
                        saved += len(rows)

            # periodic progress log (time-based)
            now = time.time()
            if (now - last_log) >= max(1.0, float(log_every)):
                elapsed = now - start
                rate = saved / max(1e-6, elapsed)
                remaining = N - saved
                eta_s = remaining / max(1e-6, rate) if rate > 0 else float("inf")
                eta_m, eta_s_rem = divmod(int(eta_s), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                log.info(
                    f"... {saved}/{N} embeddings | {rate:.1f} imgs/s | "
                    f"elapsed {elapsed/60:.1f} min | ETA {eta_h:02d}:{eta_m:02d}:{eta_s_rem:02d}"
                )
                last_log = now

    elapsed = time.time() - start
    rate = saved / max(1e-6, elapsed)
    log.info(f"round done: {saved}/{N} embedded in {elapsed/60:.1f} min ({rate:.1f} imgs/s)")

    # small optimize/housekeeping
    try:
        conn.execute("PRAGMA optimize;")
    except Exception:
        pass
    conn.close()

    return {
        "processed": float(processed),
        "saved": float(saved),
        "elapsed_s": float(elapsed),
        "throughput_imgs_per_s": float(rate),
    }


def backfill_polling(
    db_path: str,
    batch: int = 192,
    io_workers: int = 12,
    decode_mode: str = "auto",
    jpeg_reduce: int = 8,
    log_every: int = 5,
    poll_interval: int = 60,
    fetch_limit: int = 5000,
    round_limit: Optional[int] = None,
) -> None:
    """Poll forever (or up to `round_limit` rounds) for missing embeddings."""
    emb = VisionEmbedder(model_name="mobilenet_v3_large", proj_dim=128, seed=42)
    device = emb.device if emb.available() else "unavailable"
    log.info(
        f"Polling-Backfill started: batch={batch}, io_workers={io_workers}, "
        f"decode={decode_mode}, jpeg_reduce={jpeg_reduce}, device={device}"
    )

    rounds = 0
    try:
        while True:
            stats = backfill_once(
                db_path=db_path,
                batch=batch,
                io_workers=io_workers,
                decode_mode=decode_mode,
                jpeg_reduce=jpeg_reduce,
                log_every=log_every,
                fetch_limit=fetch_limit,
            )
            if stats["saved"] == 0:
                log.info(f"No pending items. Sleeping {poll_interval}s …")
                time.sleep(max(1, int(poll_interval)))
            rounds += 1
            if round_limit is not None and rounds >= int(round_limit):
                log.info("Round limit reached. Exiting.")
                break
    except KeyboardInterrupt:
        log.info("Interrupted by user. Exiting gracefully.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Backfill deep embeddings into SQLite DB")
    ap.add_argument("--db", type=str, default="image_recommender.db", help="SQLite DB path")
    ap.add_argument("--batch", type=int, default=192, help="Embedding batch size")
    ap.add_argument("--io-workers", type=int, default=12, help="Decode threads")
    ap.add_argument(
        "--decode-mode",
        type=str,
        choices=["auto", "turbo", "imdecode", "opencv"],
        default="auto",
        help="Decoder selection",
    )
    ap.add_argument("--jpeg-reduce", type=int, choices=[1, 2, 4, 8], default=8, help="JPEG reduced decode factor")
    ap.add_argument("--log-every", type=int, default=5, help="Seconds between progress logs")
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds to wait when nothing to do")
    ap.add_argument("--fetch-limit", type=int, default=5000, help="Max rows per round")
    ap.add_argument("--once", action="store_true", help="Run only a single round")
    ap.add_argument("--round-limit", type=int, default=None, help="Stop after N rounds (polling mode)")
    return ap


def main():
    ap = _build_cli()
    args = ap.parse_args()

    if args.once:
        stats = backfill_once(
            db_path=args.db,
            batch=args.batch,
            io_workers=args.io_workers,
            decode_mode=args.decode_mode,
            jpeg_reduce=args.jpeg_reduce,
            log_every=args.log_every,
            fetch_limit=args.fetch_limit,
        )
        # print compact JSON summary to stdout
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}, ensure_ascii=False))
    else:
        backfill_polling(
            db_path=args.db,
            batch=args.batch,
            io_workers=args.io_workers,
            decode_mode=args.decode_mode,
            jpeg_reduce=args.jpeg_reduce,
            log_every=args.log_every,
            poll_interval=args.poll_interval,
            fetch_limit=args.fetch_limit,
            round_limit=args.round_limit,
        )


if __name__ == "__main__":
    main()