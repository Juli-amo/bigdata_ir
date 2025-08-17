
"""
Image Recommender CLI

Subcommands:
  - ingest       : index images (features -> SQLite) using the fast bulk pipeline
  - query        : top-k search for one query image
  - query-multi  : top-k search for multiple query images (score fusion)
  - stats        : show database stats

Notes:
  - 'ingest' expects a DIRECTORY path (fast, parallelized). Single-file ingest is intentionally
    not supported here to keep the pipeline optimized and consistent with ImageDatabase.
  - Feature weights (color/embedding/custom) are normalized to sum to 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

# Local modules
from ImageDatabase import ImageDatabase
from ImageRecommender import ImageRecommender

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------------------- helpers ---------------------------------------
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Normalize non-negative weights to sum to 1; fallback to deep-only if zeros."""
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        return {"color": 0.0, "embedding": 1.0, "custom": 0.0}
    return {k: float(max(0.0, v)) / s for k, v in w.items()}


def iter_images(root: Path) -> Iterable[Path]:
    """Yield images under a directory (recursively), or the file itself if supported."""
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
        yield root
        return
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def _safe_path_from_result(r: Dict[str, Any]) -> str:
    """Best-effort extraction of a file path from a result item."""
    meta = r.get("metadata") or {}
    return meta.get("filepath") or meta.get("path") or ""


def _show_grid(title: str, images: List[Path], titles: List[str], out_path: Optional[str], show: bool) -> None:
    """Render a grid with matplotlib; optionally save to disk."""
    try:
        import cv2
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Visualization not available (missing cv2/matplotlib): {e}", file=sys.stderr)
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


# ----------------------------- commands --------------------------------------
def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the fast bulk ingest (directory only)."""
    root = Path(args.images).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Path not found: {root}", file=sys.stderr)
        sys.exit(1)
    if not root.is_dir():
        print(f"[ERROR] --images must be a DIRECTORY (got a file). "
              f"Place the file into a directory and pass the directory path.", file=sys.stderr)
        sys.exit(2)

    db = ImageDatabase(db_path=args.db)
    t0 = time.time()
    stats = db.bulk_add_directory(
        directory=str(root),
        recursive=True,
        workers=args.workers,
        pool=args.pool,
        batch_size=args.batch_size,
        num_bins=args.bins,
        k_clusters=args.k,
        resize_max=args.resize,
        decode_mode=args.decode_mode,
        jpeg_reduce=args.jpeg_reduce,
        use_file_hash=args.use_file_hash,
        log_every=args.log_every,
    )
    elapsed = time.time() - t0
    out = {
        "elapsed_sec": round(elapsed, 2),
        "ingest": stats,
        "db": db.get_database_stats() if hasattr(db, "get_database_stats") else {},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_query(args: argparse.Namespace) -> None:
    """Top-k search for a single query image."""
    db = ImageDatabase(db_path=args.db)
    rec = ImageRecommender(database=db, ann_threshold=args.ann_threshold)

    rec.weights = normalize_weights({
        "color": args.w_color,
        "embedding": args.w_embedding,
        "custom": args.w_custom,
    })

    t0 = time.time()
    results = rec.find_similar_images(args.image, top_k=args.topk, candidates=args.candidates)
    elapsed = time.time() - t0
    print(f"[INFO] Query finished in {elapsed:.3f}s")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.show or args.save_grid:
        paths = [Path(args.image)] + [Path(_safe_path_from_result(r)) for r in results]
        titles = ["QUERY"] + [
            f"{(r.get('metadata') or {}).get('filename','?')} ({r.get('similarity_score',0):.2f})"
            for r in results
        ]
        _show_grid("Query & Top-k Results", paths, titles, args.save_grid, args.show)


def cmd_query_multi(args: argparse.Namespace) -> None:
    """Top-k search for multiple query images (score fusion)."""
    db = ImageDatabase(db_path=args.db)
    rec = ImageRecommender(database=db, ann_threshold=args.ann_threshold)

    rec.weights = normalize_weights({
        "color": args.w_color,
        "embedding": args.w_embedding,
        "custom": args.w_custom,
    })

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
        titles = [f"QUERY {i+1}" for i in range(len(args.images))] + [
            f"{(r.get('metadata') or {}).get('filename','?')} ({r.get('similarity_score',0):.2f})"
            for r in results
        ]
        _show_grid("Multi-Query & Top-k Results", paths, titles, args.save_grid, args.show)


def cmd_stats(args: argparse.Namespace) -> None:
    """Print DB stats as JSON."""
    db = ImageDatabase(db_path=args.db)
    stats = db.get_database_stats() if hasattr(db, "get_database_stats") else {}
    print(json.dumps(stats, indent=2, ensure_ascii=False))


# ----------------------------- parser ----------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cli.py", description="Image Recommender CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest (DIRECTORY ONLY; uses fast bulk pipeline)
    sp = sub.add_parser("ingest", help="Index images into SQLite (directory only, fast bulk pipeline)")
    sp.add_argument("--images", required=True, help="Directory path (recursive)")
    sp.add_argument("--db", default="image_recommender.db", help="SQLite DB path")
    sp.add_argument("--workers", type=int, default=None, help="Number of workers (auto if omitted)")
    sp.add_argument("--pool", type=str, choices=["thread", "process"], default="thread", help="Worker pool type")
    sp.add_argument("--decode-mode", type=str, choices=["auto", "turbo", "imdecode", "opencv"], default="auto")
    sp.add_argument("--jpeg-reduce", type=int, choices=[1, 2, 4, 8], default=8)
    sp.add_argument("--batch-size", type=int, default=2000)
    sp.add_argument("--bins", type=int, default=32)
    sp.add_argument("--k", type=int, default=3, help="dominant colors (k-means)")
    sp.add_argument("--resize", type=int, default=256, help="max side for feature extraction")
    sp.add_argument("--use-file-hash", action="store_true", help="also compute MD5 of file content (slower)")
    sp.add_argument("--log-every", type=int, default=1000, help="progress log interval")
    sp.set_defaults(func=cmd_ingest)

    # query (single)
    sp = sub.add_parser("query", help="Search: one query image (top-k)")
    sp.add_argument("--image", required=True, help="Path to the query image")
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--candidates", type=int, default=200, help="Candidate pool size for re-ranking")
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--ann-threshold", type=int, default=1000, help="switch to ANN above this DB size")
    sp.add_argument("--w-color", type=float, default=0.4)
    sp.add_argument("--w-embedding", type=float, default=0.4)
    sp.add_argument("--w-custom", type=float, default=0.2)
    sp.add_argument("--show", action="store_true", help="Show matplotlib grid")
    sp.add_argument("--save-grid", dest="save_grid", default=None, help="Path to save a results grid (PNG)")
    sp.set_defaults(func=cmd_query)

    # query-multi
    sp = sub.add_parser("query-multi", help="Search: multiple query images (scores combined)")
    sp.add_argument("--images", nargs="+", required=True, help="List of image paths")
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--candidates", type=int, default=200)
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--ann-threshold", type=int, default=1000)
    sp.add_argument("--w-color", type=float, default=0.4)
    sp.add_argument("--w-embedding", type=float, default=0.4)
    sp.add_argument("--w-custom", type=float, default=0.2)
    sp.add_argument("--show", action="store_true")
    sp.add_argument("--save-grid", dest="save_grid", default=None)
    sp.set_defaults(func=cmd_query_multi)

    # stats
    sp = sub.add_parser("stats", help="Show database stats")
    sp.add_argument("--db", default="image_recommender.db")
    sp.set_defaults(func=cmd_stats)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()