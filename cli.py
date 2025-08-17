"""
Image Recommender CLI:
- ingest: index images into SQLite (metadata + features)
- query: top-k search for a single query image
- query-multi: combine scores from multiple query images
- stats: show database stats
Optional:
- --show: open a window with the query and result images
- --save-grid: save a montage (PNG) of query/results
"""

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

# Local modules
from ImageDatabase import ImageDatabase
from ImageRecommender import ImageRecommender

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    """Yield images under a directory (recursively), or the file itself if it's a single image."""
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
        yield root
        return
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Index (ingest) images: read files, compute features, persist to SQLite.
    Accepts a directory OR a single image path.
    """
    root = Path(args.images).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    db = ImageDatabase(db_path=args.db)
    added = 0
    errors = 0

    for img_path in iter_images(root):
        try:
            res = db.add_image(str(img_path))
            added += 1 if res is not None else 0
        except Exception as e:
            errors += 1
            print(f"[WARN] Could not ingest {img_path}: {e}", file=sys.stderr)

    stats = db.get_database_stats() if hasattr(db, "get_database_stats") else {}
    print(json.dumps({"added": added, "errors": errors, "stats": stats}, indent=2, ensure_ascii=False))


def _safe_path_from_result(r: Dict[str, Any]) -> str:
    """Best-effort extraction of a file path from a result item."""
    meta = r.get("metadata") or {}
    return meta.get("filepath") or meta.get("path") or ""


def _show_grid(title: str, images: List[Path], titles: List[str], out_path: Optional[str], show: bool) -> None:
    """Render a grid with matplotlib; if GUI not available, optionally save to disk."""
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
        elif not out_path:
            print("[INFO] No --show or --save-grid given; nothing to display.")

        plt.close()
    except Exception as e:
        print(f"[WARN] Could not display/save grid: {e}", file=sys.stderr)


def _show_results_single(query_path: str, results: List[Dict[str, Any]], out_path: Optional[str], show: bool) -> None:
    """Build a grid for a single-query search: [QUERY] + top-k results."""
    paths = [Path(query_path)] + [Path(_safe_path_from_result(r)) for r in results]
    titles = ["QUERY"] + [f"{(r.get('metadata') or {}).get('filename','?')} ({r.get('similarity_score',0):.2f})" for r in results]
    _show_grid("Query & Top-k Results", paths, titles, out_path, show)


def _show_results_multi(query_paths: List[str], results: List[Dict[str, Any]], out_path: Optional[str], show: bool) -> None:
    """Build a grid for multi-query: first all QUERIES, then top-k results."""
    query_paths_p = [Path(p) for p in query_paths]
    result_paths_p = [Path(_safe_path_from_result(r)) for r in results]
    paths = query_paths_p + result_paths_p
    query_titles = [f"QUERY {i+1}" for i in range(len(query_paths))]
    result_titles = [f"{(r.get('metadata') or {}).get('filename','?')} ({r.get('similarity_score',0):.2f})" for r in results]
    titles = query_titles + result_titles
    _show_grid("Multi-Query & Top-k Results", paths, titles, out_path, show)


def cmd_query(args: argparse.Namespace) -> None:
    """Top-k search for a single query image; optional visualization."""
    db = ImageDatabase(db_path=args.db)
    rec = ImageRecommender(database=db)
    results = rec.find_similar_images(args.image, top_k=args.topk)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.show or args.save_grid:
        _show_results_single(args.image, results, args.save_grid, args.show)


def cmd_query_multi(args: argparse.Namespace) -> None:
    """Top-k search for multiple query images; optional visualization."""
    db = ImageDatabase(db_path=args.db)
    rec = ImageRecommender(database=db)
    results = rec.find_similar_multi_input(args.images, top_k=args.topk, combination_method="average")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.show or args.save_grid:
        _show_results_multi(args.images, results, args.save_grid, args.show)


def cmd_stats(args: argparse.Namespace) -> None:
    """Print DB stats (if your ImageDatabase implements get_database_stats)."""
    db = ImageDatabase(db_path=args.db)
    stats = db.get_database_stats() if hasattr(db, "get_database_stats") else {}
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cli.py", description="Image Recommender CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest
    sp = sub.add_parser("ingest", help="Index images into the SQLite DB (metadata + features)")
    sp.add_argument("--images", required=True, help="Path to a directory OR a single image file")
    sp.add_argument("--db", default="image_recommender.db", help="SQLite DB path")
    sp.set_defaults(func=cmd_ingest)

    # query (single)
    sp = sub.add_parser("query", help="Search: one query image (top-k)")
    sp.add_argument("--image", required=True, help="Path to the query image")
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--show", action="store_true", help="Open a window showing query and result images")
    sp.add_argument("--save-grid", dest="save_grid", default=None, help="Optional path to save a results grid (e.g., out.png)")
    sp.set_defaults(func=cmd_query)

    # query-multi
    sp = sub.add_parser("query-multi", help="Search: multiple query images (scores combined)")
    sp.add_argument("--images", nargs="+", required=True, help="List of image paths")
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--db", default="image_recommender.db")
    sp.add_argument("--show", action="store_true", help="Open a window showing query and result images")
    sp.add_argument("--save-grid", dest="save_grid", default=None, help="Optional path to save a results grid (e.g., out.png)")
    sp.set_defaults(func=cmd_query_multi)

    # stats
    sp = sub.add_parser("stats", help="Show database stats")
    sp.add_argument("--db", default="image_recommender.db")
    sp.set_defaults(func=cmd_stats)

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()