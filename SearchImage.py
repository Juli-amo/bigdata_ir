import numpy as np
import pickle
import time
import json
import sqlite3
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Eigene Module
from FeatureExtraction import ColorAnalyzer, ImageFeatureExtractor
from ImageDatabase import ImageDatabase


class FeatureIndexer:
    """
    Erstellt und verwaltet Feature-Indizes für schnelle Suche
    """
    
    def __init__(self, feature_dim: int = None):
        """
        Args:
            feature_dim (int): Dimensionalität der Feature-Vektoren
        """
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.pca = None
        self.use_pca = False
        self.logger = logging.getLogger(__name__)
    
    def create_feature_matrix(self, image_database: ImageDatabase, 
                            use_pca_reduction: bool = False, 
                            pca_components: int = 50) -> Tuple[np.ndarray, List[str]]:
        """
        Erstellt Feature-Matrix aus allen Bildern in der Datenbank
        
        Args:
            image_database: ImageDatabase Instanz
            use_pca_reduction: Ob PCA für Dimensionsreduktion verwendet werden soll
            pca_components: Anzahl PCA-Komponenten
            
        Returns:
            Tuple[np.ndarray, List[str]]: (feature_matrix, image_ids)
        """
        start_time = time.time()
        
        # Alle Image IDs holen
        image_ids = image_database.get_all_image_ids()
        self.logger.info(f"Erstelle Feature-Matrix für {len(image_ids)} Bilder")
        
        if not image_ids:
            return np.array([]), []
        
        # Feature-Vektoren sammeln
        feature_vectors = []
        valid_image_ids = []
        
        for image_id in image_ids:
            features = self._extract_comprehensive_features(image_database, image_id)
            if features is not None:
                feature_vectors.append(features)
                valid_image_ids.append(image_id)
        
        if not feature_vectors:
            self.logger.warning("Keine gültigen Feature-Vektoren gefunden!")
            return np.array([]), []
        
        # Feature-Matrix erstellen
        feature_matrix = np.array(feature_vectors)
        self.logger.info(f"Feature-Matrix Shape: {feature_matrix.shape}")
        
        # Normalisierung
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Optional: PCA für große Datenmengen
        if use_pca_reduction and feature_matrix.shape[1] > pca_components:
            self.pca = PCA(n_components=pca_components)
            feature_matrix = self.pca.fit_transform(feature_matrix)
            self.use_pca = True
            self.logger.info(f"PCA Reduktion: {feature_matrix.shape[1]} Komponenten")
        
        creation_time = time.time() - start_time
        self.logger.info(f"Feature-Matrix erstellt in {creation_time:.2f}s")
        
        return feature_matrix, valid_image_ids
    
    def _extract_comprehensive_features(self, image_database: ImageDatabase, image_id: str) -> Optional[np.ndarray]:
        """
        Extrahiert umfassende Features für einen Bild-ID
        
        Args:
            image_database: ImageDatabase Instanz
            image_id: Image ID
            
        Returns:
            Optional[np.ndarray]: Feature-Vektor oder None
        """
        try:
            # Color Features aus DB
            color_features = image_database.get_color_features(image_id)
            if not color_features:
                return None
            
            # Advanced Features aus DB
            with sqlite3.connect(image_database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT texture_features FROM advanced_features WHERE image_id = ?", (image_id,))
                row = cursor.fetchone()
                texture_features = json.loads(row[0]) if row and row[0] else {}
            
            # Comprehensive Feature Vector erstellen
            features = []
            
            # 1. Dominante Farben (flach)
            dominant_colors = np.array(color_features['dominant_colors']).flatten()
            features.extend(dominant_colors)
            
            # 2. Color Statistics
            color_stats = color_features['color_stats']
            features.extend(color_stats['mean_bgr'])
            features.extend(color_stats['std_bgr'])
            features.append(color_stats['brightness'])
            
            # 3. HSV Histogram (komprimiert - nur Mittelwerte pro Kanal)
            hsv_hist = np.array(color_features['hsv_histogram'])
            for channel_hist in hsv_hist:
                features.append(np.mean(channel_hist))
                features.append(np.std(channel_hist))
            
            # 4. BGR Histogram (komprimiert)
            bgr_hist = np.array(color_features['bgr_histogram'])
            for channel_hist in bgr_hist:
                features.append(np.mean(channel_hist))
                features.append(np.std(channel_hist))
            
            # 5. Texture Features
            if texture_features:
                features.extend([
                    texture_features.get('mean', 0),
                    texture_features.get('std', 0),
                    texture_features.get('variance', 0)
                ])
            else:
                features.extend([0, 0, 0])
            
            # 6. Additional derived features
            # Color diversity (Variation zwischen dominanten Farben)
            if len(dominant_colors) > 3:
                color_diversity = np.std(dominant_colors.reshape(-1, 3), axis=0)
                features.extend(color_diversity)
            else:
                features.extend([0, 0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Feature-Extraction für {image_id}: {e}")
            return None
    
    def transform_query_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transformiert Query-Features mit gleicher Normalisierung/PCA wie Training
        
        Args:
            features: Raw feature vector
            
        Returns:
            np.ndarray: Transformierte Features
        """
        # Normalisierung
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # PCA falls verwendet
        if self.use_pca and self.pca:
            features_scaled = self.pca.transform(features_scaled)
        
        return features_scaled.flatten()


class ApproximateNearestNeighbor:
    """
    Implementiert verschiedene ANN-Algorithmen (Projektanforderung)
    """
    
    def __init__(self, algorithm: str = 'brute_force', **kwargs):
        """
        Args:
            algorithm: 'brute_force', 'lsh', 'tree_based', 'faiss' (wenn verfügbar)
            **kwargs: Algorithmus-spezifische Parameter
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.is_built = False
        self.logger = logging.getLogger(__name__)
        
        # Index-spezifische Attribute
        self.feature_matrix = None
        self.image_ids = None
        self.index = None
    
    def build_index(self, feature_matrix: np.ndarray, image_ids: List[str]):
        """
        Baut ANN-Index auf
        
        Args:
            feature_matrix: Normalisierte Feature-Matrix
            image_ids: Entsprechende Image IDs
        """
        start_time = time.time()
        self.feature_matrix = feature_matrix
        self.image_ids = image_ids
        
        if self.algorithm == 'brute_force':
            self._build_brute_force_index()
        elif self.algorithm == 'lsh':
            self._build_lsh_index()
        elif self.algorithm == 'tree_based':
            self._build_tree_index()
        elif self.algorithm == 'faiss':
            self._build_faiss_index()
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algorithm}")
        
        build_time = time.time() - start_time
        self.logger.info(f"{self.algorithm} Index gebaut in {build_time:.2f}s für {len(image_ids)} Bilder")
        self.is_built = True
    
    def _build_brute_force_index(self):
        """Brute Force - einfach Matrix speichern"""
        self.index = {
            'type': 'brute_force',
            'matrix': self.feature_matrix,
            'ids': self.image_ids
        }
    
    def _build_lsh_index(self):
        """Locality Sensitive Hashing (einfache Implementation)"""
        from sklearn.random_projection import SparseRandomProjection
        
        # Parameter
        n_projections = self.kwargs.get('n_projections', 10)
        n_bits = self.kwargs.get('n_bits', 8)
        
        # Random Projections für LSH
        projector = SparseRandomProjection(n_components=n_projections, random_state=42)
        projected_features = projector.fit_transform(self.feature_matrix)
        
        # Hash-Buckets erstellen
        hash_buckets = {}
        for i, features in enumerate(projected_features):
            # Einfacher Hash: Vorzeichen der Projektionen
            hash_key = tuple((features > 0).astype(int))
            
            if hash_key not in hash_buckets:
                hash_buckets[hash_key] = []
            hash_buckets[hash_key].append(i)
        
        self.index = {
            'type': 'lsh',
            'projector': projector,
            'buckets': hash_buckets,
            'matrix': self.feature_matrix,
            'ids': self.image_ids
        }
        
        self.logger.info(f"LSH: {len(hash_buckets)} Buckets erstellt")
    
    def _build_tree_index(self):
        """Tree-based Index (KDTree für kleine Dimensionen)"""
        from sklearn.neighbors import NearestNeighbors
        
        # Parameter
        algorithm = self.kwargs.get('tree_algorithm', 'ball_tree')
        leaf_size = self.kwargs.get('leaf_size', 30)
        
        # NearestNeighbors Index
        nn_index = NearestNeighbors(
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric='cosine'
        )
        nn_index.fit(self.feature_matrix)
        
        self.index = {
            'type': 'tree_based',
            'nn_index': nn_index,
            'ids': self.image_ids
        }
    
    def _build_faiss_index(self):
        """FAISS Index (falls verfügbar)"""
        try:
            import faiss
            
            # Parameter
            index_type = self.kwargs.get('index_type', 'IVF')
            nlist = self.kwargs.get('nlist', min(100, len(self.image_ids) // 10))
            
            # FAISS Index erstellen
            d = self.feature_matrix.shape[1]
            
            if index_type == 'IVF' and len(self.image_ids) > 100:
                # IVF Index für größere Datasets
                quantizer = faiss.IndexFlatIP(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist)
                index.train(self.feature_matrix.astype(np.float32))
                index.add(self.feature_matrix.astype(np.float32))
            else:
                # Flat Index für kleinere Datasets
                index = faiss.IndexFlatIP(d)
                index.add(self.feature_matrix.astype(np.float32))
            
            self.index = {
                'type': 'faiss',
                'faiss_index': index,
                'ids': self.image_ids
            }
            
        except ImportError:
            self.logger.warning("FAISS nicht verfügbar, verwende Brute Force")
            self._build_brute_force_index()
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Sucht k ähnlichste Bilder
        
        Args:
            query_vector: Query Feature-Vektor
            k: Anzahl Ergebnisse
            
        Returns:
            List[Tuple[str, float]]: [(image_id, similarity), ...]
        """
        if not self.is_built:
            raise RuntimeError("Index wurde noch nicht gebaut!")
        
        if self.algorithm == 'brute_force':
            return self._search_brute_force(query_vector, k)
        elif self.algorithm == 'lsh':
            return self._search_lsh(query_vector, k)
        elif self.algorithm == 'tree_based':
            return self._search_tree(query_vector, k)
        elif self.algorithm == 'faiss':
            return self._search_faiss(query_vector, k)
    
    def _search_brute_force(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Brute Force Suche"""
        similarities = cosine_similarity([query_vector], self.feature_matrix)[0]
        
        # Top-k Indizes
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            image_id = self.image_ids[idx]
            similarity = float(similarities[idx])
            results.append((image_id, similarity))
        
        return results
    
    def _search_lsh(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """LSH Suche"""
        # Query projizieren
        projected_query = self.index['projector'].transform([query_vector])
        query_hash = tuple((projected_query[0] > 0).astype(int))
        
        # Kandidaten aus Hash-Bucket sammeln
        candidates = self.index['buckets'].get(query_hash, [])
        
        # Falls zu wenige Kandidaten, weitere Buckets hinzufügen
        if len(candidates) < k * 2:
            # Ähnliche Hash-Keys finden (Hamming-Distanz)
            for hash_key, indices in self.index['buckets'].items():
                if hash_key != query_hash:
                    hamming_dist = sum(a != b for a, b in zip(query_hash, hash_key))
                    if hamming_dist <= 2:  # Toleranz
                        candidates.extend(indices)
                if len(candidates) >= k * 3:
                    break
        
        # Falls immer noch zu wenige, alle nehmen
        if len(candidates) < k:
            candidates = list(range(len(self.image_ids)))
        
        # Brute Force auf Kandidaten
        candidate_features = self.index['matrix'][candidates]
        similarities = cosine_similarity([query_vector], candidate_features)[0]
        
        # Top-k aus Kandidaten
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            original_idx = candidates[idx]
            image_id = self.image_ids[original_idx]
            similarity = float(similarities[idx])
            results.append((image_id, similarity))
        
        return results
    
    def _search_tree(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Tree-based Suche"""
        distances, indices = self.index['nn_index'].kneighbors([query_vector], n_neighbors=k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            image_id = self.image_ids[idx]
            # Cosine distance zu similarity umwandeln
            similarity = 1 - distances[0][i]
            results.append((image_id, float(similarity)))
        
        return results
    
    def _search_faiss(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """FAISS Suche"""
        similarities, indices = self.index['faiss_index'].search(
            query_vector.astype(np.float32).reshape(1, -1), k
        )
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0:  # FAISS kann -1 für nicht gefundene zurückgeben
                image_id = self.image_ids[idx]
                similarity = float(similarities[0][i])
                results.append((image_id, similarity))
        
        return results


class SearchEngine:
    """
    Hauptklasse für Image Search mit verschiedenen ANN-Algorithmen
    """
    
    def __init__(self, image_database: ImageDatabase, 
                 ann_algorithm: str = 'auto',
                 use_parallel: bool = True,
                 cache_features: bool = True):
        """
        Args:
            image_database: ImageDatabase Instanz
            ann_algorithm: ANN-Algorithmus ('auto', 'brute_force', 'lsh', 'tree_based', 'faiss')
            use_parallel: Parallele Verarbeitung verwenden
            cache_features: Feature-Matrix cachen
        """
        self.db = image_database
        self.use_parallel = use_parallel
        self.cache_features = cache_features
        self.logger = logging.getLogger(__name__)
        
        # Feature Indexer
        self.feature_indexer = FeatureIndexer()
        
        # Algorithmus bestimmen
        self.ann_algorithm = self._determine_algorithm(ann_algorithm)
        self.logger.info(f"Verwende ANN-Algorithmus: {self.ann_algorithm}")
        
        # ANN-Index
        self.ann_index = ApproximateNearestNeighbor(algorithm=self.ann_algorithm)
        
        # Cache
        self.feature_cache_path = "feature_cache.pkl"
        self.is_indexed = False
        
        # Index automatisch erstellen
        self.build_search_index()
    
    def _determine_algorithm(self, ann_algorithm: str) -> str:
        """Bestimmt optimalen ANN-Algorithmus basierend auf Datengröße"""
        image_count = len(self.db.get_all_image_ids())
        
        if ann_algorithm == 'auto':
            if image_count < 100:
                return 'brute_force'
            elif image_count < 1000:
                return 'tree_based'
            elif image_count < 10000:
                return 'lsh'
            else:
                return 'faiss'  # Für sehr große Datasets
        else:
            return ann_algorithm
    
    def build_search_index(self, force_rebuild: bool = False, 
                          use_pca: bool = None, pca_components: int = 50):
        """
        Baut Suchindex auf
        
        Args:
            force_rebuild: Index neu aufbauen auch wenn Cache existiert
            use_pca: PCA für Dimensionsreduktion verwenden
            pca_components: Anzahl PCA-Komponenten
        """
        start_time = time.time()
        
        # PCA automatisch bestimmen
        if use_pca is None:
            image_count = len(self.db.get_all_image_ids())
            use_pca = image_count > 1000  # Für große Datasets
        
        # Feature-Matrix laden oder erstellen
        if self.cache_features and Path(self.feature_cache_path).exists() and not force_rebuild:
            self.logger.info("Lade Feature-Matrix aus Cache...")
            with open(self.feature_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                feature_matrix = cache_data['feature_matrix']
                image_ids = cache_data['image_ids']
                self.feature_indexer = cache_data['indexer']
        else:
            self.logger.info("Erstelle neue Feature-Matrix...")
            feature_matrix, image_ids = self.feature_indexer.create_feature_matrix(
                self.db, use_pca_reduction=use_pca, pca_components=pca_components
            )
            
            # Cache speichern
            if self.cache_features:
                cache_data = {
                    'feature_matrix': feature_matrix,
                    'image_ids': image_ids,
                    'indexer': self.feature_indexer
                }
                with open(self.feature_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                self.logger.info("Feature-Matrix gecacht")
        
        # ANN-Index aufbauen
        if len(image_ids) > 0:
            self.ann_index.build_index(feature_matrix, image_ids)
            self.is_indexed = True
            
            index_time = time.time() - start_time
            self.logger.info(f"Suchindex komplett aufgebaut in {index_time:.2f}s")
        else:
            self.logger.warning("Keine Bilder zum Indexieren gefunden!")
    
    def search_similar_images(self, query_image_path: str, k: int = 5, 
                             similarity_threshold: float = 0.0) -> List[Dict]:
        """
        Sucht ähnliche Bilder für Query-Bild
        
        Args:
            query_image_path: Pfad zum Query-Bild
            k: Anzahl Ergebnisse
            similarity_threshold: Minimale Similarity
            
        Returns:
            List[Dict]: Suchergebnisse mit Metadaten
        """
        if not self.is_indexed:
            raise RuntimeError("Suchindex nicht aufgebaut!")
        
        start_time = time.time()
        
        try:
            # Query-Features extrahieren
            import cv2
            query_image = cv2.imread(query_image_path)
            if query_image is None:
                raise ValueError(f"Kann Query-Bild nicht laden: {query_image_path}")
            
            # Comprehensive Features für Query extrahieren
            color_analyzer = ColorAnalyzer()
            feature_extractor = ImageFeatureExtractor()
            
            # Gleiche Features wie im Index
            query_features = self._extract_query_features(query_image, color_analyzer, feature_extractor)
            
            # Features transformieren (gleiche Normalisierung wie Training)
            transformed_features = self.feature_indexer.transform_query_features(query_features)
            
            # ANN-Suche
            search_results = self.ann_index.search(transformed_features, k)
            
            # Ergebnisse mit Metadaten anreichern
            enriched_results = []
            for image_id, similarity in search_results:
                if similarity >= similarity_threshold:
                    metadata = self.db.get_image_metadata(image_id)
                    color_features = self.db.get_color_features(image_id)
                    
                    result = {
                        'image_id': image_id,
                        'similarity_score': similarity,
                        'metadata': metadata,
                        'dominant_colors': color_features['dominant_colors'] if color_features else None
                    }
                    enriched_results.append(result)
            
            search_time = time.time() - start_time
            self.logger.info(f"Suche abgeschlossen in {search_time:.3f}s - {len(enriched_results)} Ergebnisse")
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"Fehler bei Bildsuche: {e}")
            return []
    
    def _extract_query_features(self, image, color_analyzer, feature_extractor) -> np.ndarray:
        """Extrahiert Features für Query-Bild (gleiche Methode wie beim Index)"""
        # Gleiche Feature-Extraktion wie im FeatureIndexer
        
        # 1. Dominante Farben
        dominant_colors = color_analyzer.get_dominant_colors(image).flatten()
        
        # 2. Color Statistics
        mean_bgr = np.mean(image, axis=(0, 1))
        std_bgr = np.std(image, axis=(0, 1))
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # 3. HSV Histogram (komprimiert)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_stats = []
        for i in range(3):
            hist = cv2.calcHist([hsv_image], [i], None, [256], [0, 256])
            hsv_stats.extend([np.mean(hist), np.std(hist)])
        
        # 4. BGR Histogram (komprimiert)
        bgr_stats = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            bgr_stats.extend([np.mean(hist), np.std(hist)])
        
        # 5. Texture Features
        texture_features = feature_extractor.extract_texture_features(image)
        texture_vals = [
            texture_features.get('mean', 0),
            texture_features.get('std', 0),
            texture_features.get('variance', 0)
        ]
        
        # 6. Color Diversity
        color_diversity = np.std(dominant_colors.reshape(-1, 3), axis=0) if len(dominant_colors) > 3 else [0, 0, 0]
        
        # Alles kombinieren
        features = []
        features.extend(dominant_colors)
        features.extend(mean_bgr)
        features.extend(std_bgr)
        features.append(brightness)
        features.extend(hsv_stats)
        features.extend(bgr_stats)
        features.extend(texture_vals)
        features.extend(color_diversity)
        
        return np.array(features, dtype=np.float32)
    
    def batch_search(self, query_image_paths: List[str], k: int = 5) -> Dict[str, List[Dict]]:
        """
        Batch-Suche für mehrere Query-Bilder
        
        Args:
            query_image_paths: Liste von Query-Bild-Pfaden
            k: Anzahl Ergebnisse pro Query
            
        Returns:
            Dict[str, List[Dict]]: {query_path: [results]}
        """
        results = {}
        
        if self.use_parallel and len(query_image_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
                future_to_path = {
                    executor.submit(self.search_similar_images, path, k): path 
                    for path in query_image_paths
                }
                
                for future in future_to_path:
                    path = future_to_path[future]
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        self.logger.error(f"Fehler bei Batch-Suche für {path}: {e}")
                        results[path] = []
        else:
            # Sequential processing
            for path in query_image_paths:
                results[path] = self.search_similar_images(path, k)
        
        return results
    
    def get_search_stats(self) -> Dict:
        """Gibt Statistiken über die Search Engine zurück"""
        return {
            'algorithm': self.ann_algorithm,
            'indexed_images': len(self.ann_index.image_ids) if self.ann_index.image_ids else 0,
            'is_indexed': self.is_indexed,
            'use_parallel': self.use_parallel,
            'feature_dimensions': self.feature_indexer.feature_dim,
            'uses_pca': self.feature_indexer.use_pca,
            'cache_enabled': self.cache_features
        }
    
    def benchmark_search_performance(self, test_image_path: str, 
                                   k_values: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        Benchmark der Suchperformance
        
        Args:
            test_image_path: Test-Bild für Benchmark
            k_values: Verschiedene k-Werte zum Testen
            
        Returns:
            Dict: Performance-Statistiken
        """
        results = {}
        
        for k in k_values:
            times = []
            
            # Mehrere Durchläufe für stabilen Durchschnitt
            for _ in range(5):
                start_time = time.time()
                search_results = self.search_similar_images(test_image_path, k)
                search_time = time.time() - start_time
                times.append(search_time)
            
            results[f'k_{k}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'results_count': len(search_results)
            }
        
        return results


# Convenience-Funktionen
def create_search_engine(image_directory: str, 
                        ann_algorithm: str = 'auto',
                        db_path: str = "image_recommender.db") -> SearchEngine:
    """
    Erstellt komplette Search Engine
    
    Args:
        image_directory: Pfad zu Bildverzeichnis
        ann_algorithm: ANN-Algorithmus
        db_path: Pfad zur Datenbank
        
    Returns:
        SearchEngine: Initialisierte Search Engine
    """
    # Database laden oder erstellen
    from ImageDatabase import initialize_database_with_images
    db = initialize_database_with_images(image_directory, db_path)
    
    # Search Engine erstellen
    search_engine = SearchEngine(db, ann_algorithm=ann_algorithm)
    
    print("=== Search Engine bereit! ===")
    print(f"Statistiken: {search_engine.get_search_stats()}")
    
    return search_engine