# Fast image ingest into SQLite (WAL). Primary JPEG decode via pyTurboJPEG with compressed-domain scaling.
# Falls back to OpenCV decoders. Thread/Process pools supported. Rich metrics & logging.

from __future__ import annotations

import os
import sqlite3
import json
import hashlib
import datetime
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from FeatureExtraction import ColorAnalyzer, compute_phash


__all__ = ["ImageDatabase"]

# ----------------------------- logging ---------------------------------------
log = logging.getLogger("ImageDatabase")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ----------------------------- constants -------------------------------------
ALLOWED_EXTS = {".jpg", ".jpeg"}  # could be extended: ".png", ".bmp", ".tiff", ".webp"
JPEG_EXTS = {".jpg", ".jpeg"}

# ----------------------------- TurboJPEG -------------------------------------
HAVE_TURBO = False
try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
    HAVE_TURBO = True
except Exception:
    HAVE_TURBO = False


class _TLS:
    """Thread-local store for TurboJPEG handle."""
    turbo: Optional["TurboJPEG"] = None
    turbo_init_failed: bool = False


TLS = _TLS()


def _get_turbo() -> Optional["TurboJPEG"]:
    """Return (and lazily init) a TurboJPEG handle if available, else None."""
    if not HAVE_TURBO or TLS.turbo_init_failed:
        return None
    if TLS.turbo is not None:
        return TLS.turbo

    # Try explicit env var first, then common install paths (macOS/Homebrew, Windows, Linux).
    candidates = [
        os.environ.get("TURBOJPEG"),
        "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib",  # macOS (Apple Silicon)
        "/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib",     # macOS (Intel)
        "/opt/local/lib/libturbojpeg.dylib",                    # macOS (MacPorts)
        "/usr/lib/libturbojpeg.so",                             # Linux
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so",            # Linux (Debian/Ubuntu)
        r"C:\libjpeg-turbo\bin\turbojpeg.dll",                  # Windows
    ]
    try:
        lib = next((p for p in candidates if p and os.path.exists(p)), None)
        TLS.turbo = TurboJPEG(lib) if lib else TurboJPEG()  # falls back to system paths
        return TLS.turbo
    except Exception as e:
        log.info(f"TurboJPEG not available ({e}); using fallback decoders.")
        TLS.turbo_init_failed = True
        return None


# ----------------------------- helpers ---------------------------------------
def _now_iso() -> str:
    """Current timestamp in ISO format (seconds)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def _fmt_dur(sec: float) -> str:
    """Format seconds into HH:MM:SS."""
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _init_worker() -> None:
    """Reduce nested threading (OpenCV/BLAS) in worker processes for stability."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _file_hash_stream(path: str, block_size: int = 1 << 20) -> str:
    """MD5 over file contents (streaming)."""
    h = hashlib.md5()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bytes(path: str) -> Optional[bytes]:
    """Read file into bytes; return None on error."""
    try:
        with open(path, "rb", buffering=0) as f:
            return f.read()
    except Exception:
        return None


def _decode_turbo(path: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """Decode JPEG via TurboJPEG using compressed-domain scaling (1/1, 1/2, 1/4, 1/8). Returns BGR."""
    tj = _get_turbo()
    if tj is None:
        return None
    data = _read_bytes(path)
    if data is None:
        return None
    # Map 1|2|4|8 to scaling factors accepted by TurboJPEG
    sf = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}.get(int(jpeg_reduce), (1, 8))
    try:
        img = tj.decode(
            data,
            pixel_format=TJPF_BGR,
            scaling_factor=sf,
            flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT,
        )
        return img
    except Exception:
        return None


def _decode_imdecode(path: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """Decode via cv2.imdecode; optionally post-scale JPEG to approx. 1/reduce."""
    data = _read_bytes(path)
    if data is None:
        return None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and Path(path).suffix.lower() in JPEG_EXTS and jpeg_reduce in (2, 4, 8):
            f = 1.0 / float(jpeg_reduce)
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def _decode_opencv(path: str, jpeg_reduce: int) -> Optional[np.ndarray]:
    """OpenCV imread with JPEG reduced decode (IMREAD_REDUCED_COLOR_X) where available."""
    try:
        flag = cv2.IMREAD_COLOR
        ext = Path(path).suffix.lower()
        if ext in JPEG_EXTS and jpeg_reduce in (2, 4, 8):
            flag = getattr(cv2, f"IMREAD_REDUCED_COLOR_{jpeg_reduce}")
        return cv2.imread(path, flag)
    except Exception:
        return None


def _decode_image(path: str, decode_mode: str, jpeg_reduce: int) -> Tuple[Optional[np.ndarray], str]:
    """Try decoders according to `decode_mode`. Return (BGR image, decoder_used)."""
    # Skip resource forks / junk files
    name = Path(path).name
    if name.startswith("._") or name == ".DS_Store":
        return None, "skip"

    ext = Path(path).suffix.lower()

    if decode_mode == "turbo":
        if ext in JPEG_EXTS:
            img = _decode_turbo(path, jpeg_reduce)
            return img, ("turbo" if img is not None else "fail")
        img = _decode_opencv(path, jpeg_reduce)
        return img, ("opencv" if img is not None else "fail")

    if decode_mode == "imdecode":
        img = _decode_imdecode(path, jpeg_reduce)
        return img, ("imdecode" if img is not None else "fail")

    if decode_mode == "opencv":
        img = _decode_opencv(path, jpeg_reduce)
        return img, ("opencv" if img is not None else "fail")

    # auto: turbo -> imdecode -> opencv
    if ext in JPEG_EXTS and HAVE_TURBO:
        img = _decode_turbo(path, jpeg_reduce)
        if img is not None:
            return img, "turbo"
    img = _decode_imdecode(path, jpeg_reduce)
    if img is not None:
        return img, "imdecode"
    img = _decode_opencv(path, jpeg_reduce)
    return img, ("opencv" if img is not None else "fail")


# ----------------------------- worker ----------------------------------------
def _process_one(
    path: str,
    num_bins: int,
    k_clusters: int,
    resize_max: int,
    decode_mode: str,
    jpeg_reduce: int,
    use_file_hash: bool,
) -> Optional[Tuple[Tuple, Tuple, Tuple, Dict[str, float], str, str]]:
    """Worker: decode -> resize -> color features -> pHash. Returns DB rows + stats."""
    t0 = time.time()
    stats = {"io_decode": 0.0, "resize": 0.0, "color": 0.0, "phash": 0.0, "total": 0.0}
    try:
        s = time.time()
        img, d_used = _decode_image(path, decode_mode, jpeg_reduce)
        if img is None:
            return None
        stats["io_decode"] += time.time() - s

        h0, w0 = img.shape[:2]
        ch = 1 if img.ndim == 2 else img.shape[2]

        # Optional resize for feature extraction
        s = time.time()
        max_side = max(h0, w0)
        if resize_max and max_side > resize_max:
            scale = float(resize_max) / float(max_side)
            nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
            img_feat = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            img_feat = img
        stats["resize"] += time.time() - s

        # Stable image id from absolute path (fast)
        image_id = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
        fhash = _file_hash_stream(path) if use_file_hash else None

        # Color features
        s = time.time()
        ca = ColorAnalyzer(k_clusters=k_clusters, random_state=42)
        cf = ca.extract_color_features(img_feat, num_bins=num_bins, num_dominant=k_clusters)
        stats["color"] += time.time() - s

        # pHash (on resized image)
        s = time.time()
        ph = compute_phash(img_feat)
        stats["phash"] += time.time() - s

        st = os.stat(path)
        meta = (
            image_id,
            os.path.basename(path),
            os.path.abspath(path),
            int(st.st_size),
            int(w0),
            int(h0),
            int(ch),
            fhash,
            None,
            datetime.datetime.fromtimestamp(st.st_ctime).isoformat(timespec="seconds"),
            _now_iso(),
        )

        color_row = (
            image_id,
            json.dumps(cf["dominant_colors"]),
            json.dumps(cf["hsv_histogram"]),
            json.dumps(cf["bgr_histogram"]),
            json.dumps(cf["color_stats"]),
        )

        custom = {"phash_hex": hex(ph)}
        adv_row = (image_id, json.dumps(custom))  # (image_id, custom_features)

        stats["total"] = time.time() - t0
        return meta, color_row, adv_row, stats, d_used, Path(path).suffix.lower()
    except Exception as e:
        # Keep pipeline robust; just log and continue
        log.warning(f"[worker-error] {path}: {e}")
        return None


# ----------------------------- DB wrapper ------------------------------------
class ImageDatabase:
    """Relational store for images & features (SQLite/WAL). Fast bulk ingest with pools and TurboJPEG."""

    def __init__(self, db_path: str = "image_recommender.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        # Performance PRAGMAs
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA mmap_size=3000000000;")
        cur.execute("PRAGMA cache_size=-200000;")
        cur.execute("PRAGMA busy_timeout=60000;")
        self.conn.commit()
        self._create_tables()
        log.info(f"Image database initialized: {db_path}")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self.conn.close()
        except Exception:
            pass

    # ----------------------------- schema ------------------------------------
    def _create_tables(self) -> None:
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id     TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                filepath     TEXT NOT NULL,
                file_size    INTEGER,
                width        INTEGER,
                height       INTEGER,
                channels     INTEGER,
                file_hash    TEXT UNIQUE,
                photographer TEXT,
                created_date TEXT,
                added_to_db  TEXT,
                UNIQUE(filepath)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS color_features (
                image_id        TEXT PRIMARY KEY,
                dominant_colors TEXT,
                hsv_histogram   TEXT,
                bgr_histogram   TEXT,
                color_stats     TEXT,
                FOREIGN KEY (image_id) REFERENCES images (image_id)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS advanced_features (
                image_id        TEXT PRIMARY KEY,
                texture_features TEXT,
                deep_embeddings  BLOB,
                custom_features  TEXT,
                deep_dim         INTEGER,
                FOREIGN KEY (image_id) REFERENCES images (image_id)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_images_path ON images(filepath);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);")
        self.conn.commit()

    # ----------------------------- getters -----------------------------------
    def get_all_image_ids(self) -> List[str]:
        """Return all image_ids."""
        c = self.conn.cursor()
        c.execute("SELECT image_id FROM images")
        return [r[0] for r in c.fetchall()]

    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata row from images table as dict."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        c = con.cursor()
        c.execute("SELECT * FROM images WHERE image_id=?", (image_id,))
        row = c.fetchone()
        con.close()
        return dict(row) if row else None

    def get_color_features(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Return color features for an image_id as dict."""
        con = sqlite3.connect(self.db_path)
        c = con.cursor()
        c.execute(
            "SELECT dominant_colors,hsv_histogram,bgr_histogram,color_stats FROM color_features WHERE image_id=?",
            (image_id,),
        )
        row = c.fetchone()
        con.close()
        if not row:
            return None
        return {
            "dominant_colors": json.loads(row[0]),
            "hsv_histogram": json.loads(row[1]),
            "bgr_histogram": json.loads(row[2]),
            "color_stats": json.loads(row[3]),
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Return counts & DB file size (MB)."""
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM images")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM color_features")
        cf = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM advanced_features")
        af = c.fetchone()[0]
        size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            "total_images": total,
            "color_features_count": cf,
            "advanced_features_count": af,
            "database_size_mb": size / (1024 * 1024),
        }

    # ----------------------------- ingest ------------------------------------
    def bulk_add_directory(
        self,
        directory: str,
        recursive: bool = True,
        workers: Optional[int] = None,
        pool: str = "thread",
        batch_size: int = 2000,
        num_bins: int = 32,
        k_clusters: int = 3,
        resize_max: int = 256,
        decode_mode: str = "auto",
        jpeg_reduce: int = 8,
        use_file_hash: bool = False,
        log_every: int = 500,
    ) -> Dict[str, Any]:
        """Bulk ingest a directory of images with parallel decode + feature extraction."""
        root = Path(directory)
        if not root.exists():
            log.error(f"Directory not found: {directory}")
            return {"added": 0, "errors": 0}

        # Collect files (skip resource forks)
        it = root.rglob("*") if recursive else root.iterdir()
        files: List[str] = []
        for p in it:
            if not p.is_file():
                continue
            if p.name.startswith("._") or p.name == ".DS_Store":
                continue
            if p.suffix.lower() in ALLOWED_EXTS:
                files.append(str(p))

        n_files = len(files)
        n_jpeg = sum(1 for f in files if Path(f).suffix.lower() in JPEG_EXTS)
        n_other = n_files - n_jpeg
        log.info(
            f"Scanning {directory}: {n_files} files (jpeg={n_jpeg}, other={n_other}), "
            f"decode={decode_mode}, jpeg_reduce={jpeg_reduce}, pool={pool}"
        )
        if n_files == 0:
            return {"added": 0, "errors": 0}

        # Worker defaults
        cpu = os.cpu_count() or 8
        if pool == "thread":
            if workers is None:
                workers = min(64, 2 * cpu)
        else:
            if workers is None:
                workers = min(16, cpu)

        # SQL upserts
        SQL_IMG = """
            INSERT INTO images (image_id, filename, filepath, file_size, width, height, channels, file_hash, photographer, created_date, added_to_db)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                filename=excluded.filename, filepath=excluded.filepath, file_size=excluded.file_size,
                width=excluded.width, height=excluded.height, channels=excluded.channels, file_hash=excluded.file_hash
        """
        SQL_COLOR = """
            INSERT INTO color_features (image_id, dominant_colors, hsv_histogram, bgr_histogram, color_stats)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                dominant_colors=excluded.dominant_colors, hsv_histogram=excluded.hsv_histogram,
                bgr_histogram=excluded.bgr_histogram, color_stats=excluded.color_stats
        """
        SQL_ADV = """
            INSERT INTO advanced_features (image_id, texture_features, deep_embeddings, custom_features, deep_dim)
            VALUES (?, NULL, NULL, ?, NULL)
            ON CONFLICT(image_id) DO UPDATE SET custom_features=excluded.custom_features
        """

        # Timers & counters
        acc = {"io_decode": 0.0, "resize": 0.0, "color": 0.0, "phash": 0.0, "total": 0.0}
        nstat = 0
        errors = 0
        used_decoder_counts = {"turbo": 0, "imdecode": 0, "opencv": 0, "skip": 0}
        ext_counts = {"jpeg": 0, "other": 0}

        t0 = time.time()
        cur = self.conn.cursor()
        cur.execute("BEGIN")
        buf_img: List[Tuple] = []
        buf_color: List[Tuple] = []
        buf_adv: List[Tuple] = []

        Executor = ThreadPoolExecutor if pool == "thread" else ProcessPoolExecutor
        init = None if pool == "thread" else _init_worker

        try:
            with Executor(max_workers=workers, initializer=init) as ex:
                futs = [
                    ex.submit(
                        _process_one,
                        p,
                        num_bins,
                        k_clusters,
                        resize_max,
                        decode_mode,
                        jpeg_reduce,
                        use_file_hash,
                    )
                    for p in files
                ]

                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    if not res:
                        errors += 1
                        continue
                    meta, color_row, adv_row, stats, d_used, ext = res

                    # Counters
                    used_decoder_counts[d_used] = used_decoder_counts.get(d_used, 0) + 1
                    if ext in JPEG_EXTS:
                        ext_counts["jpeg"] += 1
                    else:
                        ext_counts["other"] += 1

                    buf_img.append(meta)
                    buf_color.append(color_row)
                    buf_adv.append(adv_row)
                    for k in acc:
                        acc[k] += stats[k]
                    nstat += 1

                    if len(buf_img) >= batch_size:
                        cur.executemany(SQL_IMG, buf_img)
                        cur.executemany(SQL_COLOR, buf_color)
                        cur.executemany(SQL_ADV, buf_adv)
                        buf_img.clear()
                        buf_color.clear()
                        buf_adv.clear()
                        self.conn.commit()
                        cur.execute("BEGIN")

                    if log_every and (i % log_every == 0):
                        now = time.time()
                        elapsed_total = now - t0
                        imgs_s = nstat / max(1e-6, elapsed_total)
                        remaining = max(0, n_files - nstat)
                        eta_s = remaining / max(1e-6, imgs_s)
                        eta_str = _fmt_dur(eta_s)
                        elapsed_str = _fmt_dur(elapsed_total)
                        eta_clock = (datetime.datetime.now() + datetime.timedelta(seconds=eta_s)).strftime("%H:%M:%S")

                        ms = {k: (acc[k] / max(1, nstat)) * 1000.0 for k in acc}
                        log.info(
                            f"... {nstat}/{n_files} | {imgs_s:.1f} imgs/s | "
                            f"elapsed {elapsed_str} | ETA {eta_str} (~{eta_clock}) | "
                            f"avg ms/img: io {ms['io_decode']:.1f} | resize {ms['resize']:.1f} | "
                            f"color {ms['color']:.1f} | phash {ms['phash']:.2f} | total {ms['total']:.1f}"
                        )

                # Flush remaining buffers
                if buf_img:
                    cur.executemany(SQL_IMG, buf_img)
                if buf_color:
                    cur.executemany(SQL_COLOR, buf_color)
                if buf_adv:
                    cur.executemany(SQL_ADV, buf_adv)

            self.conn.commit()
        except Exception as e:
            log.error(f"Bulk ingest failed: {e}")
            self.conn.rollback()
        finally:
            try:
                cur.execute("PRAGMA optimize;")
            except Exception:
                pass

        elapsed = time.time() - t0
        ms = {k: (acc[k] / max(1, nstat)) * 1000.0 for k in acc}
        thr = nstat / max(1e-6, elapsed)

        log.info(
            "Bulk ingest done: added=%d, errors=%d | avg ms/img: io %.1f | resize %.1f | color %.1f | phash %.2f | "
            "total %.1f | throughput %.1f imgs/s | elapsed %.1fs | workers=%s pool=%s decode=%s jpeg_reduce=%s | "
            "decoders %s | formats %s",
            nstat,
            errors,
            ms["io_decode"],
            ms["resize"],
            ms["color"],
            ms["phash"],
            ms["total"],
            thr,
            elapsed,
            workers,
            pool,
            decode_mode,
            jpeg_reduce,
            used_decoder_counts,
            ext_counts,
        )

        return {
            "added": int(nstat),
            "errors": int(errors),
            "avg_ms_per_step": ms,
            "throughput_imgs_per_s": thr,
            "elapsed_s": elapsed,
            "decoders": used_decoder_counts,
            "formats": ext_counts,
        }

    # -------------------- embeddings API (used by backfill) -------------------
    def save_deep_embeddings(self, rows: List[Tuple[str, bytes, int]]) -> None:
        """Upsert deep embeddings as BLOBs (image_id, blob, dim)."""
        cur = self.conn.cursor()
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
        self.conn.commit()


# ----------------------------- CLI -------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fast bulk ingest of images into SQLite (features+pHash).")
    ap.add_argument("--ingest", type=str, required=True, help="Root directory (recursive).")
    ap.add_argument("--db", type=str, default="image_recommender.db", help="SQLite DB path.")
    ap.add_argument("--workers", type=int, default=None, help="Number of workers (auto if omitted).")
    ap.add_argument("--pool", type=str, choices=["thread", "process"], default="thread", help="Thread vs process pool.")
    ap.add_argument("--decode-mode", type=str, choices=["auto", "turbo", "imdecode", "opencv"], default="auto")
    ap.add_argument("--jpeg-reduce", type=int, choices=[1, 2, 4, 8], default=8, help="Turbo/OpenCV reduced decode factor.")
    ap.add_argument("--batch-size", type=int, default=2000, help="DB batch size for executemany().")
    ap.add_argument("--bins", type=int, default=32, help="Histogram bins per channel.")
    ap.add_argument("--k", type=int, default=3, help="Number of dominant colors (k-means).")
    ap.add_argument("--resize", type=int, default=256, help="Max side for feature extraction (pixels).")
    ap.add_argument("--use-file-hash", action="store_true", help="Compute MD5 of file contents (slower).")
    ap.add_argument("--log-every", type=int, default=1000, help="Progress log interval (#images).")
    args = ap.parse_args()

    db = ImageDatabase(args.db)
    stats = db.bulk_add_directory(
        directory=args.ingest,
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
    print("Done:", stats, "| DB:", db.get_database_stats())