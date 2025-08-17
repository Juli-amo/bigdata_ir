import time
import json
import logging
import sqlite3
from typing import List, Dict, Tuple, Optional, Union

import cv2
import numpy as np

from FeatureExtraction import ColorAnalyzer, ImageFeatureExtractor, compute_phash
from ImageDatabase import ImageDatabase
from DeepEmbedding import VisionEmbedder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ImageRecommender")


def _infer_db_color_params(db_path: str) -> tuple[int, int]:
    """Read num_bins and k from DB (first row); fallback to (32, 3)."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT dominant_colors, hsv_histogram FROM color_features LIMIT 1")
        row = cur.fetchone()
        con.close()
        if not row:
            return 32, 3
        dom = json.loads(row[0])       # list of Kx3
        hsv = json.loads(row[1])       # [[H],[S],[V]] len=num_bins
        bins = len(hsv[0]) if isinstance(hsv, list) and hsv and isinstance(hsv[0], list) else 32
        k = len(dom) if isinstance(dom, list) else 3
        return int(bins), int(k)
    except Exception:
        return 32, 3


class SimilarityCalculator:
    """Small collection of similarity functions (color, embedding, custom)."""

    def __init__(self):
        """Init helper feature extractors used for query features."""
        self.color_analyzer = ColorAnalyzer()
        self.feature_extractor = ImageFeatureExtractor()
        self.logger = logging.getLogger(__name__)

    def _cos(self, a, b) -> float:
        """Cosine similarity with simple numeric guards."""
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _phash_sim(self, hex1: Optional[str], hex2: Optional[str], bits: int = 64) -> float:
        """Map Hamming distance of 64-bit pHash to [0,1] similarity."""
        try:
            def _to_int(x):
                if x is None:
                    return None
                if isinstance(x, int):
                    return int(x)
                if isinstance(x, str):
                    s = x.strip().lower()
                    if s.startswith("0x"):
                        s = s[2:]
                    if not s or any(c not in "0123456789abcdef" for c in s):
                        return None
                    return int(s, 16)
                return None

            a = _to_int(hex1)
            b = _to_int(hex2)
            if a is None or b is None:
                return 0.0
            x = a ^ b
            dist = x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")
            return max(0.0, 1.0 - dist / float(bits))
        except Exception:
            return 0.0

    def color_sim(self, f1: Dict, f2: Dict) -> float:
        """Weighted cosine on HSV/BGR histograms + brightness proximity."""
        try:
            hsv1 = np.array(f1['hsv_histogram']).flatten()
            hsv2 = np.array(f2['hsv_histogram']).flatten()
            bgr1 = np.array(f1['bgr_histogram']).flatten()
            bgr2 = np.array(f2['bgr_histogram']).flatten()

            hsv_sim = self._cos(hsv1, hsv2)
            bgr_sim = self._cos(bgr1, bgr2)

            bright1 = float(f1['color_stats']['brightness'])
            bright2 = float(f2['color_stats']['brightness'])
            bright_sim = max(0.0, 1.0 - abs(bright1 - bright2) / 255.0)

            s = 0.45 * hsv_sim + 0.45 * bgr_sim + 0.10 * bright_sim
            return float(min(1.0, max(0.0, s)))
        except Exception as e:
            self.logger.error(f"Color similarity error: {e}")
            return 0.0

    def emb_sim(
        self,
        e1: Optional[np.ndarray],
        e2: Optional[np.ndarray],
        fallback1: Dict,
        fallback2: Dict
    ) -> float:
        """Cosine on deep embeddings; falls back to simple texture stats."""
        if e1 is not None and e2 is not None:
            return float((self._cos(e1, e2) + 1.0) * 0.5)  # [-1,1] → [0,1]
        t1 = (fallback1 or {}).get('texture_features') or {}
        t2 = (fallback2 or {}).get('texture_features') or {}
        if not t1 or not t2:
            return 0.5
        v1 = [t1.get('mean', 0), t1.get('std', 0), t1.get('variance', 0)]
        v2 = [t2.get('mean', 0), t2.get('std', 0), t2.get('variance', 0)]
        return float((self._cos(v1, v2) + 1.0) * 0.5)

    def custom_sim(
        self,
        adv1: Dict, adv2: Dict,
        cf1: Dict, cf2: Dict,
        ph1: Optional[str], ph2: Optional[str]
    ) -> float:
        """Blend of pHash similarity, variance similarity, and entropy proximity."""
        try:
            v1 = np.array(cf1['color_stats']['std_bgr'], dtype=np.float32)
            v2 = np.array(cf2['color_stats']['std_bgr'], dtype=np.float32)
            var_sim = 1.0 - np.linalg.norm(v1 - v2) / (255.0 * np.sqrt(3.0))
            var_sim = float(min(1.0, max(0.0, var_sim)))

            def _ent(hlist):
                h = np.concatenate(hlist).astype(np.float32)
                h = h / (h.sum() + 1e-8)
                return float(-np.sum(np.where(h > 0, h * np.log(h + 1e-8), 0.0)))

            e1 = _ent(cf1['bgr_histogram'])
            e2 = _ent(cf2['bgr_histogram'])
            ent_sim = 1.0 - abs(e1 - e2) / max(e1, e2, 1.0)
            ent_sim = float(min(1.0, max(0.0, ent_sim)))

            ph_sim = self._phash_sim(ph1, ph2)
            s = 0.5 * ph_sim + 0.25 * var_sim + 0.25 * ent_sim
            return float(min(1.0, max(0.0, s)))
        except Exception:
            return 0.0


class ApproximateNearestNeighbor:
    """
    HNSW (hnswlib) with safe fallback to NumPy cosine.
    - Builds HNSW only if N > use_ann_threshold
    - Ensures ef_search >= k for queries
    """

    def __init__(
        self,
        use_ann_threshold: int = 5000,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 200
    ):
        """Store HNSW parameters and allocate holders for data/index."""
        self.use_ann_threshold = int(use_ann_threshold)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search_default = int(ef_search)
        self.ef_search_current = int(ef_search)

        self.image_ids: List[str] = []
        self.X: Optional[np.ndarray] = None
        self._use_hnsw = False
        self._norms: Optional[np.ndarray] = None
        self.name = "cheap"
        self.ann = None  # hnswlib.Index, if available

    def build(self, image_ids: List[str], X: np.ndarray, name: str):
        """Build ANN or linear index for the provided vectors."""
        t0 = time.time()
        self.name = name
        self.image_ids = list(image_ids)
        self.X = np.asarray(X, dtype=np.float32)
        Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-8)

        if len(image_ids) > self.use_ann_threshold:
            try:
                import hnswlib
                dim = int(self.X.shape[1])
                self.ann = hnswlib.Index(space='cosine', dim=dim)
                self.ann.init_index(
                    max_elements=self.X.shape[0],
                    ef_construction=self.ef_construction,
                    M=self.M
                )
                self.ann.add_items(Xn, np.arange(Xn.shape[0]))
                self.ann.set_ef(self.ef_search_default)
                self.ef_search_current = self.ef_search_default
                self._use_hnsw = True
                log.info(f"[{name}] hnswlib index built (N={self.X.shape[0]}, d={dim}, "
                         f"M={self.M}, efC={self.ef_construction}, efS={self.ef_search_current}) "
                         f"in {time.time()-t0:.2f}s")
            except Exception as e:
                self._use_hnsw = False
                log.warning(f"[{name}] hnswlib unavailable: {e} -> NumPy fallback")

        if not self._use_hnsw:
            # Precompute norms for fast cosine
            self._norms = np.linalg.norm(self.X, axis=1).astype(np.float32) + 1e-8
            log.info(f"[{name}] NumPy-cosine index (N={self.X.shape[0]}) built in {time.time()-t0:.2f}s")

    def knn(self, q: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Return top-k (image_id, cosine_sim) for query vector q."""
        if self.X is None or len(self.image_ids) == 0:
            return []

        q = np.asarray(q, dtype=np.float32).ravel()
        N = len(self.image_ids)
        k = min(max(1, int(k)), N)

        if self._use_hnsw and self.ann is not None:
            qn = q / (np.linalg.norm(q) + 1e-8)
            if k > self.ef_search_current:
                # Ensure ef_search >= k
                try:
                    self.ann.set_ef(k)
                    self.ef_search_current = k
                    log.debug(f"[{self.name}] set_ef({k}) (raised for query)")
                except Exception as e:
                    log.debug(f"[{self.name}] set_ef({k}) failed: {e}")

            labels, dists = self.ann.knn_query(qn, k=k)
            sims = 1.0 - np.clip(dists[0], 0.0, 1.0)  # cosine distance -> sim
            idxs = labels[0]
            return [(self.image_ids[int(idxs[j])], float(sims[j])) for j in range(len(idxs))]

        # Linear cosine (NumPy)
        qn = np.linalg.norm(q) + 1e-8
        sims = (self.X @ q) / (qn * self._norms)  # self._norms already +eps
        order = np.argsort(sims)[::-1][:k]
        return [(self.image_ids[i], float(sims[i])) for i in order]


class ImageRecommender:
    """
    High-level image recommender combining cheap features and deep embeddings.
    Builds both indices and supports single- and multi-query search + re-ranking.
    """

    def __init__(
        self,
        database: ImageDatabase,
        weights: Optional[Dict[str, float]] = None,
        color_bins: Optional[int] = None,
        k_colors: Optional[int] = None,
        ann_threshold: int = 1000,
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200
    ):
        """Wire DB, infer color params, init indices and embedder."""
        self.db = database
        inferred_bins, inferred_k = _infer_db_color_params(self.db.db_path)
        self.color_bins = color_bins or inferred_bins
        self.k_colors = k_colors or inferred_k

        self.sim = SimilarityCalculator()
        self.weights = weights or {'color': 0.4, 'embedding': 0.4, 'custom': 0.2}
        self.embedder = VisionEmbedder(model_name="mobilenet_v3_large", proj_dim=128, seed=42)

        self.idx_cheap = ApproximateNearestNeighbor(
            use_ann_threshold=ann_threshold,
            M=hnsw_M, ef_construction=hnsw_ef_construction, ef_search=hnsw_ef_search
        )
        self.idx_deep = ApproximateNearestNeighbor(
            use_ann_threshold=ann_threshold,
            M=hnsw_M, ef_construction=hnsw_ef_construction, ef_search=hnsw_ef_search
        )
        self._build_indices()

    def _row_to_cheap(self, row) -> Optional[Tuple[str, np.ndarray, Optional[str]]]:
        """Row -> (image_id, cheap_vector, phash_hex)."""
        try:
            iid = row[0]
            dom = np.array(json.loads(row[1]), dtype=np.float32).flatten()
            stats = json.loads(row[2])
            vstats = [stats['brightness'], *stats['mean_bgr'], *stats['std_bgr']]
            vec = list(dom) + vstats
            # texture placeholders (3 simple moments)
            if row[3]:
                try:
                    t = json.loads(row[3])
                    vec += [float(t.get('mean', 0)), float(t.get('std', 0)), float(t.get('variance', 0))]
                except Exception:
                    vec += [0.0, 0.0, 0.0]
            else:
                vec += [0.0, 0.0, 0.0]
            ph = None
            if row[4]:
                try:
                    cf = json.loads(row[4])
                    if isinstance(cf, dict):
                        ph = cf.get('phash_hex')
                except Exception:
                    pass
            return iid, np.asarray(vec, dtype=np.float32), ph
        except Exception:
            return None

    def _iter_rows(self):
        """Yield joined rows of images + color_features + advanced_features."""
        con = sqlite3.connect(self.db.db_path)
        c = con.cursor()
        c.execute(
            """
            SELECT i.image_id, cf.dominant_colors, cf.color_stats,
                   af.texture_features, af.custom_features,
                   af.deep_embeddings, af.deep_dim
            FROM images i
            JOIN color_features cf ON cf.image_id = i.image_id
            LEFT JOIN advanced_features af ON af.image_id = i.image_id
            """
        )
        rows = c.fetchall()
        con.close()
        return rows

    def _build_indices(self):
        """Build both indices from DB content (cheap + deep)."""
        t0 = time.time()
        ids_cheap, Xc, ids_deep, Xd = [], [], [], []
        deep_count = 0

        for r in self._iter_rows():
            cheap = self._row_to_cheap(r)
            if cheap:
                iid, v, _ = cheap
                ids_cheap.append(iid)
                Xc.append(v)

            blob, dim = r[5], r[6]
            if blob is not None and dim:
                try:
                    arr = np.frombuffer(blob, dtype=np.float32)
                    if arr.size == int(dim):
                        ids_deep.append(r[0])
                        Xd.append(arr)
                        deep_count += 1
                except Exception:
                    pass

        if ids_cheap:
            self.idx_cheap.build(ids_cheap, np.asarray(Xc, dtype=np.float32), name="cheap")
        if ids_deep:
            self.idx_deep.build(ids_deep, np.asarray(Xd, dtype=np.float32), name="deep")

        log.info(
            f"Indices built in {time.time()-t0:.2f}s | cheap={len(ids_cheap)}, deep={deep_count}"
        )

    def _query_features(self, img: np.ndarray):
        """Compute query features: cheap vec, deep embedding, CF dict, ADV dict, phash hex."""
        # color features
        cf = self.sim.color_analyzer.extract_color_features(
            img, num_bins=self.color_bins, num_dominant=self.k_colors
        )
        # simple texture stats
        tf = self.sim.feature_extractor.extract_texture_features(img)

        # compact helper resize for pHash
        def _shrink_for_phash(im, max_side=256):
            h, w = im.shape[:2]
            s = max(h, w)
            if s > max_side:
                scale = max_side / float(s)
                return cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return im

        try:
            shrink = _shrink_for_phash(img, max_side=256)
            ph_hex = hex(compute_phash(shrink))
        except Exception:
            ph_hex = None

        dom = np.array(cf['dominant_colors'], dtype=np.float32).flatten()
        stats = cf['color_stats']
        vstats = [stats['brightness'], *stats['mean_bgr'], *stats['std_bgr']]
        cheap_vec = np.asarray(
            list(dom) + vstats + [tf.get('mean', 0), tf.get('std', 0), tf.get('variance', 0)],
            dtype=np.float32
        )

        # deep embedding if available
        z = self.embedder.embed_array(img) if self.embedder.available() else None
        return cheap_vec, z, cf, {'texture_features': tf}, ph_hex

    def _bulk_fetch_candidates(self, cand_ids: List[str]):
        """Fetch color/adv/embedding blobs for a candidate id list."""
        if not cand_ids:
            return []

        con = sqlite3.connect(self.db.db_path)
        c = con.cursor()
        ph = ",".join(["?"] * len(cand_ids))
        c.execute(
            f"""
            SELECT i.image_id,
                   cf.dominant_colors, cf.hsv_histogram, cf.bgr_histogram, cf.color_stats,
                   af.texture_features, af.custom_features, af.deep_embeddings, af.deep_dim
            FROM images i
            JOIN color_features cf ON cf.image_id=i.image_id
            LEFT JOIN advanced_features af ON af.image_id=i.image_id
            WHERE i.image_id IN ({ph})
            """,
            cand_ids,
        )
        rows = c.fetchall()
        con.close()
        return rows

    def _rerank(
        self,
        q_cf: Dict, q_adv: Dict,
        q_deep: Optional[np.ndarray], q_ph: Optional[str],
        rows: List[tuple],
        top_k: int
    ) -> List[Dict]:
        """Compute final scores and return top-k result dicts."""
        results = []
        for r in rows:
            iid = r[0]
            db_cf = {
                'dominant_colors': json.loads(r[1]),
                'hsv_histogram': json.loads(r[2]),
                'bgr_histogram': json.loads(r[3]),
                'color_stats': json.loads(r[4]),
            }

            db_adv = {}
            db_ph = None
            if r[5]:
                try:
                    db_adv = {'texture_features': json.loads(r[5])}
                except Exception:
                    db_adv = {}
            if r[6]:
                try:
                    cfj = json.loads(r[6])
                    if isinstance(cfj, dict):
                        db_ph = cfj.get('phash_hex')
                except Exception:
                    db_ph = None

            db_emb = None
            if r[7] is not None and r[8]:
                try:
                    emb_arr = np.frombuffer(r[7], dtype=np.float32)
                    if emb_arr.size == int(r[8]):
                        db_emb = emb_arr
                except Exception:
                    db_emb = None

            c_sim = self.sim.color_sim(q_cf, db_cf)
            e_sim = self.sim.emb_sim(q_deep, db_emb, q_adv, db_adv)
            u_sim = self.sim.custom_sim(q_adv, db_adv, q_cf, db_cf, q_ph, db_ph)

            score = (
                self.weights['color'] * c_sim +
                self.weights['embedding'] * e_sim +
                self.weights['custom'] * u_sim
            )
            results.append(
                (iid, float(score),
                 {'color_similarity': c_sim, 'embedding_similarity': e_sim, 'custom_similarity': u_sim})
            )

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        out = []
        for iid, sc, det in results:
            meta = self.db.get_image_metadata(iid)
            out.append({'image_id': iid, 'similarity_score': sc, 'detailed_scores': det, 'metadata': meta})
        return out

    def find_similar_images(
        self,
        input_image: Union[str, np.ndarray],
        top_k: int = 5,
        candidates: int = 200
    ) -> List[Dict]:
        """Single-query search: extract features → ANN/linear candidates → re-rank."""
        T = {}
        t = time.time()
        if isinstance(input_image, str):
            qimg = cv2.imread(input_image)
            if qimg is None:
                raise ValueError(f"Cannot load image: {input_image}")
        else:
            qimg = input_image
        T['load'] = time.time() - t

        t = time.time()
        q_cheap, q_deep, q_cf, q_adv, q_ph = self._query_features(qimg)
        T['features'] = time.time() - t

        # choose index
        use_deep = (len(self.idx_deep.image_ids) > 0) and (q_deep is not None)
        idx = self.idx_deep if use_deep else self.idx_cheap
        N = len(idx.image_ids)
        k = min(max(top_k * 10, candidates), max(1, N))

        t = time.time()
        nbrs = idx.knn(q_deep if use_deep else q_cheap, k)
        T['ann'] = time.time() - t
        if not nbrs:
            return []

        cand_ids = [i for i, _ in nbrs]

        t = time.time()
        rows = self._bulk_fetch_candidates(cand_ids)
        T['dbfetch'] = time.time() - t

        t = time.time()
        out = self._rerank(q_cf, q_adv, q_deep, q_ph, rows, top_k)
        T['rerank'] = time.time() - t

        log.info(
            f"Search: load={T['load']:.3f}s feat={T['features']:.3f}s ann={T['ann']:.3f}s "
            f"db={T['dbfetch']:.3f}s rerank={T['rerank']:.3f}s | "
            f"index={'deep' if use_deep else 'cheap'} N={N}"
        )
        return out

    def find_similar_multi_input(
        self,
        input_images: List[Union[str, np.ndarray]],
        top_k: int = 5,
        candidates: int = 200,
        combine: str = "mean"
    ) -> List[Dict]:
        """Multi-query search: average or max combine of query features/embeddings."""
        if not input_images:
            return []

        T = {}
        t = time.time()
        q_cheaps, q_deeps, q_cfs, q_advs, q_phs = [], [], [], [], []

        # extract all query features
        for item in input_images:
            if isinstance(item, str):
                img = cv2.imread(item)
                if img is None:
                    log.warning(f"Skipping unreadable image: {item}")
                    continue
            else:
                img = item
            c, z, cf, adv, ph = self._query_features(img)
            q_cheaps.append(c)
            if z is not None:
                q_deeps.append(z)
            q_cfs.append(cf)
            q_advs.append(adv)
            q_phs.append(ph)

        if not q_cheaps:
            return []

        # combine features
        if combine == "max":
            q_cheap = np.max(np.vstack(q_cheaps), axis=0)
            q_deep = np.max(np.vstack(q_deeps), axis=0) if q_deeps else None
        else:  # mean (default)
            q_cheap = np.mean(np.vstack(q_cheaps), axis=0)
            q_deep = np.mean(np.vstack(q_deeps), axis=0) if q_deeps else None

        # simple merger for dict-like parts (use first as representative)
        q_cf = q_cfs[0]
        q_adv = q_advs[0]
        q_ph = q_phs[0]
        T['features_multi'] = time.time() - t

        # choose index
        use_deep = (len(self.idx_deep.image_ids) > 0) and (q_deep is not None)
        idx = self.idx_deep if use_deep else self.idx_cheap
        N = len(idx.image_ids)
        k = min(max(top_k * 10, candidates), max(1, N))

        t = time.time()
        nbrs = idx.knn(q_deep if use_deep else q_cheap, k)
        T['ann'] = time.time() - t
        if not nbrs:
            return []

        cand_ids = [i for i, _ in nbrs]

        t = time.time()
        rows = self._bulk_fetch_candidates(cand_ids)
        T['dbfetch'] = time.time() - t

        t = time.time()
        out = self._rerank(q_cf, q_adv, q_deep, q_ph, rows, top_k)
        T['rerank'] = time.time() - t

        log.info(
            f"Multi-search: q={len(q_cheaps)} combine={combine} feat={T['features_multi']:.3f}s "
            f"ann={T['ann']:.3f}s db={T['dbfetch']:.3f}s rerank={T['rerank']:.3f}s | "
            f"index={'deep' if use_deep else 'cheap'} N={N}"
        )
        return out

    def get_system_stats(self) -> Dict:
        """Return simple index/weight stats for UI."""
        return {
            'weights': self.weights,
            'cheap_index_size': len(self.idx_cheap.image_ids),
            'deep_index_size': len(self.idx_deep.image_ids)
        }


# Convenience bootstrap
def create_recommender_system(image_directory: str, db_path: str = "image_recommender.db") -> ImageRecommender:
    """Optional helper: ingest a folder then build the recommender."""
    db = ImageDatabase(db_path)
    if image_directory:
        db.bulk_add_directory(image_directory, recursive=True)
    return ImageRecommender(db)


if __name__ == "__main__":
    IMAGE_PATH = None
    rec = create_recommender_system(image_directory="")
    if IMAGE_PATH:
        res = rec.find_similar_images(IMAGE_PATH, top_k=5)
        for i, r in enumerate(res, 1):
            print(f"{i}. {r['metadata'].get('filename','?')} (Score: {r['similarity_score']:.3f})")