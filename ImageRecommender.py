"""
Image Recommender - Hauptklasse für Big Data Image Recommender
Orchestriert alle Komponenten und erfüllt Projektanforderungen
"""

import numpy as np
import cv2
import json
from typing import List, Dict, Tuple, Optional, Union
import time
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import sqlite3

# Eigene Module importieren
from FeatureExtraction import ColorAnalyzer, ImageFeatureExtractor, compute_phash
from ImageDatabase import ImageDatabase, ImageLoader

logger = logging.getLogger("ImageRecommender")
logging.basicConfig(level=logging.INFO)



class SimilarityCalculator:
    """
    Implementiert die 3 geforderten Similarity-Measures
    """
    
    def __init__(self):
        """Initialisiert den Similarity Calculator"""
        self.color_analyzer = ColorAnalyzer()
        self.feature_extractor = ImageFeatureExtractor()
        self.logger = logging.getLogger(__name__)

    def _cosine(self, a, b) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _phash_sim(self, hex1: Optional[str], hex2: Optional[str], bits: int = 64) -> float:
        """Robuste Ähnlichkeit aus pHash (int oder '0x..' String). Gibt 0.0 zurück, wenn etwas fehlt/kaputt ist."""
        try:
            import numpy as np

            def _to_int(x):
                if x is None:
                    return None
                # bereits int?
                if isinstance(x, (int, np.integer)):
                    return int(x)
                # String?
                if isinstance(x, str):
                    s = x.strip().lower()
                    if s.startswith("0x"):
                        s = s[2:]
                    # nur echte Hex-Zeichen zulassen
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
        
    def calculate_color_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        1. Pflicht-Similarity: Color-based (nutzt deine ColorAnalyzer Klasse)
        
        Args:
            features1 (Dict): Color Features von Bild 1
            features2 (Dict): Color Features von Bild 2
            
        Returns:
            float: Color Similarity Score (0-1, höher = ähnlicher)
        """
        try:
            # Histogramme (ohne Annahmen über Bin-Anzahl)
            hsv1 = np.array(features1['hsv_histogram']).flatten()
            hsv2 = np.array(features2['hsv_histogram']).flatten()
            bgr1 = np.array(features1['bgr_histogram']).flatten()
            bgr2 = np.array(features2['bgr_histogram']).flatten()

            hsv_sim = self._cosine(hsv1, hsv2)            # 0..1
            bgr_sim = self._cosine(bgr1, bgr2)            # 0..1

            # Helligkeit (0..1)
            bright1 = float(features1['color_stats']['brightness'])
            bright2 = float(features2['color_stats']['brightness'])
            bright_sim = max(0.0, 1.0 - abs(bright1 - bright2) / 255.0)

            # Gewichtete Mischung (Dominant-Color-Teil lassen wir weg – Histos sind robuster)
            final_sim = 0.45 * hsv_sim + 0.45 * bgr_sim + 0.10 * bright_sim

            return float(min(1.0, max(0.0, final_sim)))
        except Exception as e:
            self.logger.error(f"Fehler bei Color Similarity: {e}")
            return 0.0


    """
    def calculate_color_similarity(self, features1: Dict, features2: Dict) -> float:
        ""
        1. Pflicht-Similarity: Color-based (nutzt deine ColorAnalyzer Klasse)
        
        Args:
            features1 (Dict): Color Features von Bild 1
            features2 (Dict): Color Features von Bild 2
            
        Returns:
            float: Color Similarity Score (0-1, höher = ähnlicher)
        ""
        try:
            # Dominante Farben Similarity
            colors1 = np.array(features1['dominant_colors'])
            colors2 = np.array(features2['dominant_colors'])
            color_distance = np.linalg.norm(colors1 - colors2)
            color_sim = 1 / (1 + color_distance / 100)  # Normalisierung
            
            # HSV Histogramm Similarity
            hist1 = np.array(features1['hsv_histogram']).flatten()
            hist2 = np.array(features2['hsv_histogram']).flatten()
            hist_sim = cv2.compareHist(hist1.astype(np.float32), 
                                     hist2.astype(np.float32), 
                                     cv2.HISTCMP_CORREL)
            hist_sim = max(0, hist_sim)  # Negative Werte auf 0 setzen
            
            # Brightness Similarity
            bright1 = features1['color_stats']['brightness']
            bright2 = features2['color_stats']['brightness']
            bright_sim = 1 - abs(bright1 - bright2) / 255
            
            # Gewichtete Kombination
            final_sim = 0.4 * color_sim + 0.4 * hist_sim + 0.2 * bright_sim
            
            return min(1.0, max(0.0, final_sim))
            
        except Exception as e:
            self.logger.error(f"Fehler bei Color Similarity: {e}")
            return 0.0
        """
    
    def calculate_embedding_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        2. Pflicht-Similarity: Deep Learning Embeddings
        (Aktuell Placeholder - kann später mit echten DL-Embeddings ersetzt werden)
        
        Args:
            features1 (Dict): Advanced Features von Bild 1
            features2 (Dict): Advanced Features von Bild 2
            
        Returns:
            float: Embedding Similarity Score (0-1)
        """
        try:
            # Für jetzt: Texture-based Pseudo-Embeddings
            texture1 = features1.get('texture_features', {})
            texture2 = features2.get('texture_features', {})
            
            if not texture1 or not texture2:
                return 0.5  # Neutral wenn keine Features
            
            # Feature-Vektor aus Texture Features erstellen
            vec1 = [texture1.get('mean', 0), texture1.get('std', 0), texture1.get('variance', 0)]
            vec2 = [texture2.get('mean', 0), texture2.get('std', 0), texture2.get('variance', 0)]
            
            # Cosine Similarity
            vec1 = np.array(vec1).reshape(1, -1)
            vec2 = np.array(vec2).reshape(1, -1)
            
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            # Normalisierung auf [0, 1]
            return (similarity + 1) / 2
            
        except Exception as e:
            self.logger.error(f"Fehler bei Embedding Similarity: {e}")
            return 0.0
        
    def calculate_custom_similarity(self, features1: Dict, features2: Dict, 
                                  color_features1: Dict, color_features2: Dict, 
                                  phash_hex1: Optional[str] = None, 
                                  phash_hex2: Optional[str] = None) -> float:  
        try:
            # Varianz-Ähnlichkeit der BGR-Std
            var1 = np.array(color_features1['color_stats']['std_bgr'], dtype=np.float32)
            var2 = np.array(color_features2['color_stats']['std_bgr'], dtype=np.float32)
            var_sim = 1.0 - np.linalg.norm(var1 - var2) / (255.0 * np.sqrt(3.0))
            var_sim = float(min(1.0, max(0.0, var_sim)))

            # Entropie-Ähnlichkeit (Gleichmäßigkeit der BGR-Verteilung)
            def _entropy(hist_list):
                h = np.concatenate(hist_list).astype(np.float32)
                h = h / (h.sum() + 1e-8)
                return float(-np.sum(np.where(h > 0, h * np.log(h + 1e-8), 0.0)))
            entropy1 = _entropy(color_features1['bgr_histogram'])
            entropy2 = _entropy(color_features2['bgr_histogram'])
            entropy_sim = 1.0 - abs(entropy1 - entropy2) / max(entropy1, entropy2, 1.0)
            entropy_sim = float(min(1.0, max(0.0, entropy_sim)))

            # Texture-Komplexität (falls vorhanden)
            texture1 = (features1 or {}).get('texture_features') or {}
            texture2 = (features2 or {}).get('texture_features') or {}
            if texture1 and texture2:
                c1 = float(texture1.get('std', 0.0)) / (float(texture1.get('mean', 1.0)) + 1.0)
                c2 = float(texture2.get('std', 0.0)) / (float(texture2.get('mean', 1.0)) + 1.0)
                complexity_sim = 1.0 - abs(c1 - c2)
                complexity_sim = float(min(1.0, max(0.0, complexity_sim)))
            else:
                complexity_sim = 0.5  # neutral

            # pHash-Ähnlichkeit (robust gegen fehlende Werte)
            phash_sim = self._cosine([0, 1], [0, 1]) * 0.0  # dummy init
            phash_sim = self._phash_sim(phash_hex1, phash_hex2)

            # Mischung – gib pHash mehr Gewicht (duplikat-/motivnah)
            final_sim = 0.5 * phash_sim + 0.25 * var_sim + 0.25 * entropy_sim
            return float(min(1.0, max(0.0, final_sim)))
        except Exception as e:
            self.logger.error(f"Fehler bei Custom Similarity: {e}")
            return 0.0 

    """
    def calculate_custom_similarity(self, features1: Dict, features2: Dict, 
                                  color_features1: Dict, color_features2: Dict, 
                                  phash_hex1: Optional[str] = None, 
                                  phash_hex2: Optional[str] = None) -> float:
        ""
        3. Frei wählbare Similarity: Multi-Modal Approach
        Kombiniert verschiedene Aspekte für robuste Ähnlichkeit
        
        Args:
            features1/2 (Dict): Advanced Features
            color_features1/2 (Dict): Color Features
            
        Returns:
            float: Custom Similarity Score (0-1)
        ""
        try:
            # Aspect 1: Color Variance Similarity
            var1 = np.array(color_features1['color_stats']['std_bgr'])
            var2 = np.array(color_features2['color_stats']['std_bgr'])
            var_sim = 1 - np.linalg.norm(var1 - var2) / (255 * np.sqrt(3))
            
            # Aspect 2: Texture Complexity Similarity
            texture1 = features1.get('texture_features', {})
            texture2 = features2.get('texture_features', {})
            
            if texture1 and texture2:
                complexity1 = texture1.get('std', 0) / (texture1.get('mean', 1) + 1)
                complexity2 = texture2.get('std', 0) / (texture2.get('mean', 1) + 1)
                complexity_sim = 1 - abs(complexity1 - complexity2)
            else:
                complexity_sim = 0.5
            
            # Aspect 3: Color Distribution Uniformity
            hist1 = np.array(color_features1['bgr_histogram'])
            hist2 = np.array(color_features2['bgr_histogram'])
            
            # Berechne Entropie als Maß für Gleichmäßigkeit
            def calculate_entropy(hist):
                hist_flat = np.concatenate(hist)
                hist_flat = hist_flat + 1e-8  # Avoid log(0)
                return -np.sum(hist_flat * np.log(hist_flat))
            
            entropy1 = calculate_entropy(hist1)
            entropy2 = calculate_entropy(hist2)
            entropy_sim = 1 - abs(entropy1 - entropy2) / max(entropy1, entropy2, 1)
            
            # Gewichtete Kombination
            final_sim = 0.4 * var_sim + 0.3 * complexity_sim + 0.3 * entropy_sim
            
            return min(1.0, max(0.0, final_sim))
            
        except Exception as e:
            self.logger.error(f"Fehler bei Custom Similarity: {e}")
            return 0.0
        """


class ApproximateNearestNeighbor:
    """
    Implementiert ANN für schnelle Suche (Projektanforderung)
    Für kleine Datasets: Brute Force, für große: echte ANN-Algorithmen
    """
    
    def __init__(self, use_ann_threshold: int = 1000):
        """
        Args:
            use_ann_threshold (int): Ab dieser Anzahl Bilder echte ANN verwenden
        """
        self.use_ann_threshold = use_ann_threshold
        self.logger = logging.getLogger(__name__)
    
    def build_index(self, image_ids: List[str], feature_vectors: List[np.ndarray]):
        """
        Baut Index für schnelle Suche auf
        
        Args:
            image_ids (List[str]): Liste der Image IDs
            feature_vectors (List[np.ndarray]): Feature-Vektoren
        """
        self.image_ids = image_ids
        self.feature_vectors = np.array(feature_vectors)
        
        if len(image_ids) > self.use_ann_threshold:
            self.logger.info("Große Datenbank - würde echte ANN-Bibliothek verwenden")
            # Hier könnte FAISS, Annoy, oder ähnliches integriert werden
            # Für jetzt: Optimierte Brute Force
        else:
            self.logger.info(f"Kleine Datenbank ({len(image_ids)} Bilder) - verwende Brute Force")
    
    def find_nearest_neighbors(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Findet k ähnlichste Bilder
        
        Args:
            query_vector (np.ndarray): Query Feature-Vektor
            k (int): Anzahl zu findender Nachbarn
            
        Returns:
            List[Tuple[str, float]]: [(image_id, similarity_score), ...]
        """
        if len(self.feature_vectors) == 0:
            return []
        
        # Cosine Similarity für alle Vektoren berechnen
        similarities = cosine_similarity([query_vector], self.feature_vectors)[0]
        
        # Top-k Indizes finden
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Ergebnisse zusammenstellen
        results = []
        for idx in top_k_indices:
            image_id = self.image_ids[idx]
            similarity = similarities[idx]
            results.append((image_id, float(similarity)))
        
        return results


class ImageRecommender:
    """
    Hauptklasse - orchestriert alles (Projektanforderung erfüllt)
    """
    
    def __init__(self, database: ImageDatabase, weights: Dict[str, float] = None):
        """
        Initialisiert den Image Recommender
        
        Args:
            database (ImageDatabase): Referenz zur Image Database
            weights (Dict): Gewichtung der Similarity-Measures
        """
        self.db = database
        self.similarity_calc = SimilarityCalculator()
        self.ann = ApproximateNearestNeighbor()
        
        # Standard-Gewichtung der 3 Similarity-Measures
        self.weights = weights or {
            'color': 0.4,        # Color-based Similarity
            'embedding': 0.3,    # Deep Learning Embeddings  
            'custom': 0.3        # Custom Multi-Modal
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Index beim Start aufbauen
        self._build_search_index()
    
    def _build_search_index(self):
        """Baut Suchindex für alle Bilder in der Datenbank auf"""
        start_time = time.time()
        
        # Alle Image IDs holen
        image_ids = self.db.get_all_image_ids()
        
        if not image_ids:
            self.logger.warning("Keine Bilder in der Datenbank gefunden!")
            return
        
        # Feature-Vektoren für ANN erstellen
        feature_vectors = []
        valid_image_ids = []
        
        for image_id in image_ids:
            features = self._extract_combined_features(image_id)
            if features is not None:
                feature_vectors.append(features)
                valid_image_ids.append(image_id)
        
        # ANN Index aufbauen
        if feature_vectors:
            self.ann.build_index(valid_image_ids, feature_vectors)
            build_time = time.time() - start_time
            self.logger.info(f"Suchindex für {len(valid_image_ids)} Bilder aufgebaut ({build_time:.2f}s)")
        else:
            self.logger.error("Keine gültigen Feature-Vektoren gefunden!")
    
    def _extract_combined_features(self, image_id: str) -> Optional[np.ndarray]:
        """
        Extrahiert kombinierten Feature-Vektor für ein Bild
        
        Args:
            image_id (str): Image ID
            
        Returns:
            Optional[np.ndarray]: Kombinierter Feature-Vektor oder None
        """
        try:
            # Features aus DB laden
            color_features = self.db.get_color_features(image_id)
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT texture_features FROM advanced_features WHERE image_id = ?", (image_id,))
                row = cursor.fetchone()
                advanced_features = {'texture_features': json.loads(row[0])} if row and row[0] else None
            
            if not color_features:
                return None
            
            # Feature-Vektor erstellen
            features = []
            
            # Color Features
            dominant_colors = np.array(color_features['dominant_colors']).flatten()
            color_stats = [
                color_features['color_stats']['brightness'],
                *color_features['color_stats']['mean_bgr'],
                *color_features['color_stats']['std_bgr']
            ]
            features.extend(dominant_colors)
            features.extend(color_stats)
            
            # Texture Features
            if advanced_features and advanced_features['texture_features']:
                texture = advanced_features['texture_features']
                features.extend([texture.get('mean', 0), texture.get('std', 0), texture.get('variance', 0)])
            else:
                features.extend([0, 0, 0])  # Placeholder
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren von Features für {image_id}: {e}")
            return None
    
    def _calculate_combined_similarity(self, image_id1: str, image_id2: str) -> float:
        """
        Berechnet kombinierte Similarity zwischen zwei Bildern
        
        Args:
            image_id1 (str): Erste Image ID
            image_id2 (str): Zweite Image ID
            
        Returns:
            float: Kombinierte Similarity (0-1)
        """
        try:
            # Features laden
            color_features1 = self.db.get_color_features(image_id1)
            color_features2 = self.db.get_color_features(image_id2)
            if not color_features1 or not color_features2:
                return 0.0           
            
            # Advanced Features laden
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT texture_features, custom_features FROM advanced_features WHERE image_id IN (?, ?)", 
                             (image_id1, image_id2))
                #rows = cursor.fetchall()
                
                #advanced_features1 = {'texture_features': json.loads(rows[0][0])} if rows and rows[0][0] else {}
                #advanced_features2 = {'texture_features': json.loads(rows[1][0])} if len(rows) > 1 and rows[1][0] else {}
                row1 = cursor.fetchone()
                adv1 = {'texture_features': json.loads(row1[0])} if row1 and row1[0] else {}
                ph1 = (json.loads(row1[1]).get('phash_hex') if row1 and row1[1] else None)

                cursor.execute("SELECT texture_features, custom_features FROM advanced_features WHERE image_id = ?", (image_id2,))
                row2 = cursor.fetchone()
                adv2 = {'texture_features': json.loads(row2[0])} if row2 and row2[0] else {}
                ph2 = (json.loads(row2[1]).get('phash_hex') if row2 and row2[1] else None)


            # Alle 3 Similarity-Measures berechnen
            color_sim = self.similarity_calc.calculate_color_similarity(color_features1, color_features2)
            embedding_sim = self.similarity_calc.calculate_embedding_similarity(adv1, adv2)
            custom_sim = self.similarity_calc.calculate_custom_similarity(
                adv1, adv2, color_features1, color_features2,
                phash_hex1=ph1, phash_hex2=ph2
            )
            
            # Gewichtete Kombination
            combined_sim = (
                self.weights['color'] * color_sim +
                self.weights['embedding'] * embedding_sim +
                self.weights['custom'] * custom_sim
            )
            
            return combined_sim
            
        except Exception as e:
            self.logger.error(f"Fehler bei Similarity-Berechnung: {e}")
            return 0.0
    
    def find_similar_images(self, input_image: Union[str, np.ndarray], top_k: int = 5) -> List[Dict]:
        """
        Findet ähnlichste Bilder für ein Eingabebild (Hauptanforderung)
        
        Args:
            input_image (Union[str, np.ndarray]): Pfad zum Bild oder Bild-Array
            top_k (int): Anzahl zurückzugebender ähnlicher Bilder
            
        Returns:
            List[Dict]: Liste der ähnlichsten Bilder mit Metadaten und Scores
        """
        start_time = time.time()
        
        try:
            # Input-Bild laden falls Pfad gegeben
            if isinstance(input_image, str):
                query_image = cv2.imread(input_image)
                if query_image is None:
                    raise ValueError(f"Kann Bild nicht laden: {input_image}")
            else:
                query_image = input_image
            
            # Features für Query-Bild extrahieren
            query_color_features = self.similarity_calc.color_analyzer.extract_color_features(query_image)
            query_advanced_features = self.similarity_calc.feature_extractor.extract_texture_features(query_image)
            try:
                _q = compute_phash(query_image)
                query_phash_hex = hex(int(_q)) if _q is not None else None
            except Exception:
                query_phash_hex = None

            # Alle Bilder in DB vergleichen
            all_image_ids = self.db.get_all_image_ids()
            similarities = []
            
            for image_id in all_image_ids:
                try:
                    # Features aus DB laden
                    db_color_features = self.db.get_color_features(image_id)
                    if not db_color_features:
                        continue
                    
                    with sqlite3.connect(self.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT texture_features, custom_features FROM advanced_features WHERE image_id = ?", (image_id,))
                        row = cursor.fetchone()
                        db_advanced_features, db_phash_hex = {}, None
                        if row:
                            # texture_features
                            if row[0]:
                                try:
                                    db_advanced_features = {'texture_features': json.loads(row[0])}
                                except Exception:
                                    db_advanced_features = {}
                            # custom_features → pHash
                            if row[1]:
                                try:
                                    cf = json.loads(row[1])
                                    if isinstance(cf, dict):
                                        db_phash_hex = cf.get('phash_hex')
                                except Exception:
                                    db_phash_hex = None
                    
                    # Similarity berechnen
                    color_sim = self.similarity_calc.calculate_color_similarity(query_color_features, db_color_features)
                    embedding_sim = self.similarity_calc.calculate_embedding_similarity(query_advanced_features, db_advanced_features)
                    custom_sim = self.similarity_calc.calculate_custom_similarity(
                        query_advanced_features, db_advanced_features, query_color_features, db_color_features,
                        phash_hex1=query_phash_hex, phash_hex2=db_phash_hex
                    )
                    
                    # Kombinierte Similarity
                    combined_sim = (
                        self.weights['color'] * color_sim +
                        self.weights['embedding'] * embedding_sim +
                        self.weights['custom'] * custom_sim
                    )
                    
                    similarities.append((image_id, combined_sim, {
                        'color_similarity': color_sim,
                        'embedding_similarity': embedding_sim,
                        'custom_similarity': custom_sim
                    }))
                    
                except Exception as e:
                    self.logger.warning(f"Fehler bei Vergleich mit {image_id}: {e}")
                    continue
            
            # Top-k ähnlichste auswählen
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:top_k]
            
            # Ergebnisse mit Metadaten zusammenstellen
            results = []
            for image_id, similarity, detailed_scores in top_similarities:
                metadata = self.db.get_image_metadata(image_id)
                
                result = {
                    'image_id': image_id,
                    'similarity_score': similarity,
                    'detailed_scores': detailed_scores,
                    'metadata': metadata
                }
                results.append(result)
            
            search_time = time.time() - start_time
            self.logger.info(f"Ähnlichkeitssuche abgeschlossen in {search_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei find_similar_images: {e}")
            return []
    
    def find_similar_multi_input(self, image_list: List[Union[str, np.ndarray]], 
                                top_k: int = 5, combination_method: str = 'average') -> List[Dict]:
        """
        Findet ähnlichste Bilder für mehrere Eingabebilder (Zusatzanforderung)
        
        Args:
            image_list (List): Liste von Bildpfaden oder Bild-Arrays
            top_k (int): Anzahl zurückzugebender Bilder
            combination_method (str): Methode zur Kombination ('average', 'max', 'weighted')
            
        Returns:
            List[Dict]: Liste der ähnlichsten Bilder
        """
        if not image_list:
            return []
        
        self.logger.info(f"Multi-Input Suche mit {len(image_list)} Bildern")
        
        # Für jedes Input-Bild Similarities berechnen
        all_results = []
        for i, input_image in enumerate(image_list):
            results = self.find_similar_images(input_image, top_k=top_k*2)  # Mehr holen für bessere Kombination
            all_results.append(results)
        
        # Ergebnisse kombinieren
        combined_scores = {}
        
        for results in all_results:
            for result in results:
                image_id = result['image_id']
                similarity = result['similarity_score']
                
                if image_id not in combined_scores:
                    combined_scores[image_id] = []
                combined_scores[image_id].append(similarity)
        
        # Scores je nach Methode kombinieren
        final_scores = []
        for image_id, scores in combined_scores.items():
            if combination_method == 'average':
                final_score = np.mean(scores)
            elif combination_method == 'max':
                final_score = np.max(scores)
            elif combination_method == 'weighted':
                # Gewichtung: mehr Gewicht wenn von mehr Input-Bildern ähnlich
                weight = len(scores) / len(image_list)
                final_score = np.mean(scores) * weight
            else:
                final_score = np.mean(scores)
            
            final_scores.append((image_id, final_score))
        
        # Top-k auswählen
        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_final = final_scores[:top_k]
        
        # Ergebnisse mit Metadaten zusammenstellen
        results = []
        for image_id, final_score in top_final:
            metadata = self.db.get_image_metadata(image_id)
            
            result = {
                'image_id': image_id,
                'combined_similarity_score': final_score,
                'combination_method': combination_method,
                'metadata': metadata
            }
            results.append(result)
        
        return results
    
    def get_recommendations(self, image_id: str = None, random_seed: bool = False, 
                          top_k: int = 5) -> List[Dict]:
        """
        Allgemeine Empfehlungsfunktion
        
        Args:
            image_id (str): Basis-Bild für Empfehlungen (optional)
            random_seed (bool): Zufälliges Startbild verwenden
            top_k (int): Anzahl Empfehlungen
            
        Returns:
            List[Dict]: Liste von Empfehlungen
        """
        try:
            all_image_ids = self.db.get_all_image_ids()
            
            if not all_image_ids:
                self.logger.warning("Keine Bilder in der Datenbank!")
                return []
            
            # Basis-Bild bestimmen
            if random_seed or image_id is None:
                import random
                base_image_id = random.choice(all_image_ids)
                self.logger.info(f"Verwende zufälliges Basis-Bild: {base_image_id}")
            else:
                base_image_id = image_id
            
            # Ähnliche Bilder für Basis-Bild finden
            base_metadata = self.db.get_image_metadata(base_image_id)
            if not base_metadata:
                self.logger.error(f"Metadaten für {base_image_id} nicht gefunden")
                return []
            
            # Basis-Bild laden und Empfehlungen finden
            base_image = cv2.imread(base_metadata['filepath'])
            if base_image is None:
                self.logger.error(f"Kann Basis-Bild nicht laden: {base_metadata['filepath']}")
                return []
            
            recommendations = self.find_similar_images(base_image, top_k=top_k+1)  # +1 weil Basis-Bild dabei sein könnte
            
            # Basis-Bild aus Empfehlungen entfernen
            recommendations = [r for r in recommendations if r['image_id'] != base_image_id][:top_k]
            
            # Basis-Bild Info hinzufügen
            for rec in recommendations:
                rec['based_on_image'] = {
                    'image_id': base_image_id,
                    'filename': base_metadata['filename']
                }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Fehler bei get_recommendations: {e}")
            return []
    
    def get_system_stats(self) -> Dict:
        """Gibt Statistiken über das Recommender-System zurück"""
        db_stats = self.db.get_database_stats()
        
        return {
            'database_stats': db_stats,
            'similarity_weights': self.weights,
            'ann_threshold': self.ann.use_ann_threshold,
            'indexed_images': len(self.ann.image_ids) if hasattr(self.ann, 'image_ids') else 0
        }


# Convenience-Funktionen für einfache Verwendung
def create_recommender_system(image_directory: str, db_path: str = "image_recommender.db") -> ImageRecommender:
    """
    Erstellt komplettes Recommender-System aus Bildverzeichnis
    
    Args:
        image_directory (str): Pfad zu Bildverzeichnis
        db_path (str): Pfad zur Datenbank
        
    Returns:
        ImageRecommender: Initialisiertes System
    """
    # Database mit Bildern initialisieren
    from image_database import initialize_database_with_images
    db = initialize_database_with_images(image_directory, db_path)
    
    # Recommender erstellen
    recommender = ImageRecommender(db)
    
    print("=== Image Recommender System bereit! ===")
    print(f"Statistiken: {recommender.get_system_stats()}")
    
    return recommender


if __name__ == "__main__":
    # Beispiel für die Verwendung mit deinen 5 Bildern
    IMAGE_DIR = "/Users/juliamoor/Desktop/bigdata_ir/Bilder"  # Passe den Pfad an!
    
    # Komplettes System erstellen
    recommender = create_recommender_system(IMAGE_DIR)
    
    # Test 1: Ähnliche Bilder für erstes Bild finden
    print("\n=== Test 1: find_similar_images ===")
    all_ids = recommender.db.get_all_image_ids()
    if all_ids:
        first_image_metadata = recommender.db.get_image_metadata(all_ids[0])
        first_image_path = first_image_metadata['filepath']
        
        similar_images = recommender.find_similar_images(first_image_path, top_k=3)
        
        print(f"Ähnliche Bilder zu {first_image_metadata['filename']}:")
        for i, result in enumerate(similar_images, 1):
            print(f"  {i}. {result['metadata']['filename']} (Score: {result['similarity_score']:.3f})")
            print(f"     Details: Color={result['detailed_scores']['color_similarity']:.3f}, "
                  f"Embedding={result['detailed_scores']['embedding_similarity']:.3f}, "
                  f"Custom={result['detailed_scores']['custom_similarity']:.3f}")
    
    # Test 2: Multi-Input Suche
    print("\n=== Test 2: find_similar_multi_input ===")
    if len(all_ids) >= 2:
        # Erste zwei Bilder als Input verwenden
        input_images = []
        for i in range(min(2, len(all_ids))):
            metadata = recommender.db.get_image_metadata(all_ids[i])
            input_images.append(metadata['filepath'])
        
        multi_results = recommender.find_similar_multi_input(input_images, top_k=2)
        
        print(f"Multi-Input Suche mit {len(input_images)} Bildern:")
        for i, result in enumerate(multi_results, 1):
            print(f"  {i}. {result['metadata']['filename']} (Combined Score: {result['combined_similarity_score']:.3f})")
    
    # Test 3: Allgemeine Empfehlungen
    print("\n=== Test 3: get_recommendations ===")
    recommendations = recommender.get_recommendations(random_seed=True, top_k=2)
    
    print("Zufällige Empfehlungen:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['metadata']['filename']} (Score: {rec['similarity_score']:.3f})")
        print(f"     Basierend auf: {rec['based_on_image']['filename']}")