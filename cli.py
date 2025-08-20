"""
Image Recommender CLI

Commands:
- ingest        : index images into SQLite (features)
- query         : search with one query image
- query-multi   : search with multiple query images (score fusion)
- stats         : print DB stats
- backfill      : ONE-SHOT deep-embedding backfill (no polling)

Notes:
- The 'backfill' command decodes in parallel, embeds in batches, commits in chunks,
  and logs throughput + ETA.
"""

import argparse
import datetime
import json
import math
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from DeepEmbedding import VisionEmbedder
from ImageDatabase import ImageDatabase
from ImageRecommender import ImageRecommender

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Optional TurboJPEG (fast JPEG decode & 1/2,1/4,1/8 scaling)
try:
    from turbojpeg import TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJPF_BGR, TurboJPEG

    _TJ = TurboJPEG(os.environ.get("TURBOJPEG")) if os.environ.get("TURBOJPEG") else TurboJPEG()
except Exception:
    _TJ = None

# ---------------- helpers ----------------


def iter_images(root: Path) -> Iterable[Path]:
    """Yield images under a directory (recursively), or the file itself if it's an image."""
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:  # single file
        yield root
        return
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def _safe_path_from_result(r: dict[str, Any]) -> str:
    meta = r.get("metadata") or {}
    return meta.get("filepath") or meta.get("path") or ""


def _show_grid(
    title: str, images: list[Path], titles: list[str], out_path: Optional[str], show: bool
) -> None:
    """Render a simple grid using matplotlib (optional)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Visualization not available: {e}", file=sys.stderr)
        return

    loaded = []
    for p in images:
        if not p:
            loaded.append(None)
            continue
        img = cv2.imread(str(p))
        if img is None:
            loaded.append(None)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        loaded.append(img)

    n = len(loaded)
    if n == 0:
        print("[INFO] Nothing to display.")
        return
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)

    try:
        plt.figure(figsize=(4 * cols, 3.5 * rows))
        plt.suptitle(title)
        for i, (img, t) in enumerate(zip(loaded, titles), start=1):
            ax = plt.subplot(rows, cols, i)
            if img is not None:
                ax.imshow(img)
            ax.set_title(t)
            ax.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"[INFO] Saved grid to {out_path}")
        if show:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not display/save grid: {e}", file=sys.stderr)


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        return {"color": 0.0, "embedding": 1.0, "custom": 0.0}
    return {k: float(max(0.0, v)) / s for k, v in w.items()}


# --------------- ONE-SHOT BACKFILL helpers ---------------


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb", buffering=0) as f:
            return f.read()
    except Exception:
        return None


def _decode_one(path: str, decode_mode: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """Decode file -> BGR np.ndarray (uses TurboJPEG if available)."""
    if os.path.basename(path).startswith("._") or path.endswith(".DS_Store"):
        return None
    ext = Path(path).suffix.lower()
    if decode_mode in ("turbo", "auto") and _TJ and ext in (".jpg", ".jpeg"):
        data = _read_bytes(path)
        if data is not None:
            try:
                sf = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}.get(int(jpeg_reduce), (1, 8))
                return _TJ.decode(
                    data,
                    pixel_format=TJPF_BGR,
                    scaling_factor=sf,
                    flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT,
                )
            except Exception:
                pass
    # fallback: cv2.imdecode (+ cheap post-scale for JPEG)
    data = _read_bytes(path)
    if data is None:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None and ext in (".jpg", ".jpeg") and jpeg_reduce in (2, 4, 8):
        f = 1.0 / float(jpeg_reduce)
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA
        )
    return img


def _fmt_hms(sec: float) -> str:
    """Format seconds → HH:MM:SS."""
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _iter_missing_embeddings(db_path: str, limit: Optional[int] = None) -> list[tuple[str, str]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    q = """
    SELECT i.image_id, i.filepath
    FROM images i
    LEFT JOIN advanced_features a ON a.image_id = i.image_id
    WHERE a.deep_embeddings IS NULL OR a.deep_embeddings = X''
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = cur.execute(q).fetchall()
    con.close()
    return [(r[0], r[1]) for r in rows]


def _save_embedding_rows(db_path: str, rows: list[tuple[str, memoryview, int]]) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executemany(
        """
      INSERT INTO advanced_features (image_id, texture_features, deep_embeddings, custom_features, deep_dim)
      VALUES (?, NULL, ?, NULL, ?)
      ON CONFLICT(image_id) DO UPDATE SET
        deep_embeddings=excluded.deep_embeddings,
        deep_dim=excluded.deep_dim
    """,
        rows,
    )
    con.commit()
    con.close()


# ---------------- commands ----------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """Index images into SQLite (features)."""
    root = Path(args.images).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Path not found: {root}", file=sys.stderr)
        sys.exit(1)
    db = ImageDatabase(db_path=args.db)
    added, errors = 0, 0
    t0 = time.time()
    for img_path in iter_images(root):
        try:
            res = db.add_image(str(img_path))
            added += 1 if res is not None else 0
        except Exception as e:
            errors += 1
            print(f"[WARN] Could not ingest {img_path}: {e}", file=sys.stderr)
    elapsed = time.time() - t0
    stats = db.get_database_stats() if hasattr(db, "get_database_stats") else {}
    print(
        json.dumps(
            {"added": added, "errors": errors, "elapsed_sec": round(elapsed, 2), "stats": stats},
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_query(args: argparse.Namespace) -> None:
    """Top-k search for a single query image."""
    db = ImageDatabase(db_path=args.db)
    rec = ImageRecommender(database=db, ann_threshold=args.ann_threshold, cache_dir=args.cache_dir)
    rec.weights = _normalize_weights(
        {"color": args.w_color, "embedding": args.w_embedding, "custom": args.w_custom}
    )
    t0 = time.time()
    results = rec.find_similar_images(args.image, top_k=args.topk, candidates=args.candidates)
    elapsed = time.time() - t0
    print(f"[INFO] Query finished in {elapsed:.3f}s")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.show or args.save_grid:
        paths = [Path(args.image)] + [Path(_safe_path_from_result(r)) for r in results]
        titles = ["QUERY"] + [
            f"{(r.get('metadata') or {}).get('filename', '?')} ({r.get('similarity_score', 0):.2f})"
            for r in results
        ]
        _show_grid("Query & Top-k Results", paths, titles, args.save_grid, args.show)


def cmd_query_multi(args: argparse.Namespace) -> None:
    """Top-k search for multiple query images (score fusion)."""
    db = ImageDatabase(db_path=args.db)
    rec = ImageRecommender(database=db, ann_threshold=args.ann_threshold, cache_dir=args.cache_dir)
    rec.weights = _normalize_weights(
        {"color": args.w_color, "embedding": args.w_embedding, "custom": args.w_custom}
    )
    t0 = time.time()
    results = rec.find_similar_multi_input(
        args.images, top_k=args.topk, candidates=args.candidates, combination_method="average"
    )
    elapsed = time.time() - t0
    print(f"[INFO] Multi-Query finished in {elapsed:.3f}s")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.show or args.save_grid:
        query_paths = [Path(p) for p in args.images]
        result_paths = [Path(_safe_path_from_result(r)) for r in results]
        paths = query_paths + result_paths
        titles = [f"QUERY {i + 1}" for i in range(len(args.images))] + [
            f"{(r.get('metadata') or {}).get('filename', '?')} ({r.get('similarity_score', 0):.2f})"
            for r in results
        ]
        _show_grid("Multi-Query & Top-k Results", paths, titles, args.save_grid, args.show)


def cmd_stats(args: argparse.Namespace) -> None:
    """Print DB stats."""
    db = ImageDatabase(db_path=args.db)
    stats = db.get_database_stats() if hasattr(db, "get_database_stats") else {}
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def cmd_backfill(args: argparse.Namespace) -> None:
    """ONE-SHOT: compute all missing deep embeddings once, with timed logging."""
    # throttle BLAS threads for stability
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Best-effort PRAGMAs
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=3000000000;")
    cur.execute("PRAGMA cache_size=-200000;")
    con.commit()
    con.close()

    todo = _iter_missing_embeddings(args.db, args.limit)
    N = len(todo)
    if N == 0:
        print("[INFO] No missing embeddings. Done.")
        return

    print(
        f"[INFO] Backfill once: N={N} | batch={args.batch} | io_workers={args.io_workers} | "
        f"decode={args.decode_mode} | jpeg_reduce={args.jpeg_reduce}"
    )

    emb = VisionEmbedder(model_name="mobilenet_v3_large", proj_dim=128, seed=42)
    if not emb.available():
        print("[ERROR] VisionEmbedder/Device not available.")
        sys.exit(2)

    start_ts = time.time()
    last_log_ts = start_ts
    last_log_saved = 0
    saved = 0
    chunk_size = max(args.batch * args.io_workers, args.batch)

    def _maybe_log(force: bool = False):
        nonlocal last_log_ts, last_log_saved
        now = time.time()
        since_s = now - last_log_ts
        since_items = saved - last_log_saved
        # Log wenn: genug Items ODER genug Sekunden ODER force
        if force or since_items >= args.log_every or since_s >= args.log_seconds:
            elapsed = now - start_ts
            rate_global = saved / max(1e-6, elapsed)
            rate_chunk = since_items / max(1e-6, since_s) if since_s > 0 else 0.0
            remain = N - saved
            eta_s = int(remain / max(1e-6, rate_global)) if saved > 0 else 0
            eta_clock = (datetime.datetime.now() + datetime.timedelta(seconds=eta_s)).strftime(
                "%H:%M:%S"
            )
            print(
                f"... {saved}/{N} | {rate_global:.1f} imgs/s (global), {rate_chunk:.1f} (chunk) | "
                f"elapsed {_fmt_hms(elapsed)} | ETA {_fmt_hms(eta_s)} (~{eta_clock})",
                flush=True,
            )
            last_log_ts = now
            last_log_saved = saved

    for ci in range(0, N, chunk_size):
        chunk = todo[ci : ci + chunk_size]

        # parallel decode
        imgs = [None] * len(chunk)
        with ThreadPoolExecutor(max_workers=args.io_workers) as ex:
            futs = {
                ex.submit(_decode_one, p, args.decode_mode, args.jpeg_reduce): j
                for j, (_, p) in enumerate(chunk)
            }
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    imgs[j] = fut.result()
                except Exception:
                    imgs[j] = None

        # embed & save in mini-batches
        rows: list[tuple[str, memoryview, int]] = []
        for bi in range(0, len(chunk), args.batch):
            sub = chunk[bi : bi + args.batch]
            sub_imgs = imgs[bi : bi + args.batch]
            valid = [(k, im) for k, im in enumerate(sub_imgs) if im is not None]
            if not valid:
                _maybe_log()  # trotzdem periodisch loggen
                continue

            idx_local, imgs_valid = zip(*valid)
            Z = emb.embed_batch(list(imgs_valid))

            # Safe emptiness check for list/tuple OR numpy array
            if (
                Z is None
                or (hasattr(Z, "size") and Z.size == 0)
                or (hasattr(Z, "__len__") and len(Z) == 0)
            ):
                continue

            for off, zi in zip(idx_local, Z):
                iid = sub[off][0]
                rows.append((iid, memoryview(zi.tobytes()), int(zi.shape[0])))
                saved += 1
                # Log ggf. alle k oder alle s Sek.
                _maybe_log()

            # write out periodically (bound write batch size)
            if len(rows) >= args.log_every:
                _save_embedding_rows(args.db, rows)
                rows.clear()

        if rows:
            _save_embedding_rows(args.db, rows)
            rows.clear()
            _maybe_log(force=True)

    # final log
    _maybe_log(force=True)
    elapsed = time.time() - start_ts
    print(f"[DONE] {saved}/{N} in {_fmt_hms(elapsed)} ({saved / max(1e-6, elapsed):.1f} imgs/s)")

    # housekeeping
    os.system(f"sqlite3 {args.db} 'PRAGMA wal_checkpoint(TRUNCATE);'")
    os.system(f"sqlite3 {args.db} 'PRAGMA optimize;'")


# ---------------- parser ----------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cli.py", description="Image Recommender CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest
    sp = sub.add_parser("ingest", help="Index images into SQLite DB")
    sp.add_argument("--images", required=True, help="Directory or single image")
    sp.add_argument("--db", default="image_recommender.db")
    sp.set_defaults(func=cmd_ingest)

    # query (single)
    sp = sub.add_parser("query", help="Search with one query image")
    sp.add_argument("--image", required=True)
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument(
        "--candidates", type=int, default=200, help="Candidate pool size (for re-ranking)"
    )
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--ann-threshold", type=int, default=1000)
    sp.add_argument("--w-color", type=float, default=0.3)
    sp.add_argument("--w-embedding", type=float, default=0.5)
    sp.add_argument("--w-custom", type=float, default=0.2)
    sp.add_argument("--show", action="store_true")
    sp.add_argument("--save-grid", dest="save_grid", default=None)
    sp.add_argument("--cache-dir", default=None)
    sp.set_defaults(func=cmd_query)

    # query-multi
    sp = sub.add_parser("query-multi", help="Search with multiple query images")
    sp.add_argument("--images", nargs="+", required=True)
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--candidates", type=int, default=200)
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--ann-threshold", type=int, default=1000)
    sp.add_argument("--w-color", type=float, default=0.3)
    sp.add_argument("--w-embedding", type=float, default=0.5)
    sp.add_argument("--w-custom", type=float, default=0.2)
    sp.add_argument("--show", action="store_true")
    sp.add_argument("--save-grid", dest="save_grid", default=None)
    sp.add_argument("--cache-dir", default=None)
    sp.set_defaults(func=cmd_query_multi)

    # stats
    sp = sub.add_parser("stats", help="Show DB stats")
    sp.add_argument("--db", default="image_recommender.db")
    sp.set_defaults(func=cmd_stats)

    # backfill (ONE SHOT)
    sp = sub.add_parser("backfill", help="ONE-SHOT: compute missing deep embeddings (no polling)")
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--batch", type=int, default=320)
    sp.add_argument("--io-workers", type=int, default=24)
    sp.add_argument(
        "--decode-mode", choices=["auto", "turbo", "imdecode", "opencv"], default="auto"
    )
    sp.add_argument("--jpeg-reduce", type=int, choices=[1, 2, 4, 8], default=8)
    sp.add_argument(
        "--limit", type=int, default=None, help="Process at most N images (for testing)"
    )
    sp.add_argument("--log-every", type=int, default=2000, help="Commit+log every N vectors")
    sp.add_argument("--log-seconds", type=float, default=30.0, help="Also log every S seconds")
    sp.set_defaults(func=cmd_backfill)

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
