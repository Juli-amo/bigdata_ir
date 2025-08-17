import sqlite3, os, json, hashlib, datetime
from pathlib import Path
from typing import Generator, List, Dict, Optional, Tuple, Any
import cv2
import numpy as np
import logging

from FeatureExtraction import ColorAnalyzer, compute_phash, ImageFeatureExtractor
logger = logging.getLogger("ImageDatabase")

# Import der eigenen Feature Extraction
# from FeatureExtraction import ColorAnalyzer, 


class ImageDatabase:
    """
    Relationale Datenbank für Image Metadata und Features
    unique image-IDs, Metadaten, Dateilinks
    """
    
    def __init__(self, db_path: str = "image_recommender.db"):
        """
        Initialisiert die Image Database
        
        Args:
            db_path (str): Pfad zur SQLite Datenbank
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        self.conn.commit()
        self.color_analyzer = ColorAnalyzer()
        self.feature_extractor = ImageFeatureExtractor()
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Database initialisieren
        self._create_tables()
        self.logger.info(f"Image Database initialisiert: {db_path}")
    
    def _create_tables(self):
        """Erstellt die benötigten Tabellen"""
        #with sqlite3.connect(self.db_path) as conn:
            #cursor = conn.cursor()
        cursor = self.conn.cursor()
        # Haupttabelle für Image Metadaten
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                channels INTEGER,
                file_hash TEXT UNIQUE,
                photographer TEXT,
                created_date TEXT,
                added_to_db TEXT,
                UNIQUE(filepath)
             )
        """)    
            
            # Tabelle für Color Features (von ColorAnalyzer)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS color_features (
                image_id TEXT PRIMARY KEY,
                dominant_colors TEXT,  -- JSON serialized
                hsv_histogram TEXT,    -- JSON serialized
                bgr_histogram TEXT,    -- JSON serialized
                color_stats TEXT,      -- JSON serialized (mean, std, etc.)
                FOREIGN KEY (image_id) REFERENCES images (image_id)
            )
        """)
            
        # Tabelle für zusätzliche Features (für Texture, Deep Learning Embeddings, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advanced_features (
                image_id TEXT PRIMARY KEY,
                texture_features TEXT,     -- JSON serialized
                deep_embeddings TEXT,      -- JSON serialized (für später)
                custom_features TEXT,      -- JSON serialized
                FOREIGN KEY (image_id) REFERENCES images (image_id)
            )
        """)
            
        #conn.commit()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_path ON images(filepath);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);")
        self.conn.commit()
        self.logger.info("Database Tabellen erstellt/überprüft")

    def _now_iso(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _file_hash(self, path: str, block_size: int = 1 << 20) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def generate_image_id(self, filepath: str) -> str:
        """
        Generiert eindeutige image_id aus Dateipfad
        
        Args:
            filepath (str): Pfad zur Bilddatei
            
        Returns:
            str: Eindeutige Image ID
        """
        # Verwende MD5 Hash des Dateipfads für eindeutige ID
        return hashlib.md5(filepath.encode()).hexdigest()[:16]
    
    def extract_metadata(self, filepath: str) -> Dict:
        """
        Extrahiert Metadaten aus Bilddatei
        
        Args:
            filepath (str): Pfad zur Bilddatei
            
        Returns:
            Dict: Metadaten Dictionary
        """
        try:
            # Datei-Metadaten
            file_stats = os.stat(filepath)
            
            # Bild laden für Dimensionen
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Kann Bild nicht laden: {filepath}")
            
            # File Hash für Duplikatserkennung
            #with open(filepath, 'rb') as f:
                #file_hash = hashlib.md5(f.read()).hexdigest()

            file_hash = self._file_hash(filepath)

            metadata = {
                'filename': os.path.basename(filepath),
                'filepath': os.path.abspath(filepath),
                'file_size': file_stats.st_size,
                'width': image.shape[1],
                'height': image.shape[0],
                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                'file_hash': file_hash,
                'photographer': None,
                #'created_date': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                #'added_to_db': datetime.now().isoformat()
                'created_date': datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'added_to_db': datetime.datetime.now().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren von Metadaten für {filepath}: {e}")
            return None
    
    def extract_color_features(self, image: np.ndarray, image_id: str) -> Dict:
        """
        Extrahiert Farbfeatures mit deiner ColorAnalyzer Klasse
        
        Args:
            image (np.ndarray): Geladenes Bild
            image_id (str): Image ID
            
        Returns:
            Dict: Color Features
        """
        try:
            # Dominante Farben (aus deiner Implementierung)
            dominant_colors = self.color_analyzer.get_dominant_colors(image)
            
            # HSV Histogramm berechnen
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_hist = []
            for i in range(3):
                hist = cv2.calcHist([hsv_image], [i], None, [256], [0, 256])
                hsv_hist.append(cv2.normalize(hist, hist).flatten().tolist())
            
            # BGR Histogramm berechnen
            bgr_hist = []
            for i in range(3):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                bgr_hist.append(cv2.normalize(hist, hist).flatten().tolist())
            
            # Farbstatistiken
            color_stats = {
                'mean_bgr': np.mean(image, axis=(0, 1)).tolist(),
                'std_bgr': np.std(image, axis=(0, 1)).tolist(),
                'brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            }
            
            features = {
                'dominant_colors': dominant_colors.tolist(),
                'hsv_histogram': hsv_hist,
                'bgr_histogram': bgr_hist,
                'color_stats': color_stats
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren von Color Features für {image_id}: {e}")
            return None
    
    def extract_advanced_features(self, image: np.ndarray, image_id: str) -> Dict:
        """
        Extrahiert erweiterte Features (Texture, etc.)
        
        Args:
            image (np.ndarray): Geladenes Bild
            image_id (str): Image ID
            
        Returns:
            Dict: Advanced Features
        """
        try:
            # Texture Features
            texture_features = self.feature_extractor.extract_texture_features(image)
            
            features = {
                'texture_features': texture_features,
                'deep_embeddings': None,  # Placeholder für später
                'custom_features': None   # Placeholder für später
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren von Advanced Features für {image_id}: {e}")
            return None
    
    def add_image(self, path: str) -> Optional[str]:
        """
        Insert/Upsert an image:
        - read metadata
        - compute & persist color features (histograms + dominant colors + stats)
        - compute & persist pHash in advanced_features.custom_features
        Returns image_id (str) or None on failure.
        """
        try:
            path = os.path.abspath(path)
            if not os.path.exists(path):
                logger.warning(f"Path does not exist: {path}")
                return None

            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Cannot read image: {path}")
                return None

            # --- metadata ---
            st = os.stat(path)
            h, w = img.shape[:2]
            ch = 1 if img.ndim == 2 else img.shape[2]
            fhash = self._file_hash(path)
            fname = os.path.basename(path)
            cur = self.conn.cursor()
            cur.execute("SELECT image_id FROM images WHERE filepath=? OR file_hash=?", (path, fhash))
            row = cur.fetchone()
            if row:
                image_id = row[0]
            else:
                image_id = fhash[:16]  # stable short id
            # image_id = fhash[:16]  # stable short id

            cur = self.conn.cursor()

            # images (UPSERT by image_id)
            cur.execute("""
            INSERT INTO images (image_id, filename, filepath, file_size, width, height, channels, file_hash, photographer, created_date, added_to_db)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                filename=excluded.filename,
                filepath=excluded.filepath,
                file_size=excluded.file_size,
                width=excluded.width,
                height=excluded.height,
                channels=excluded.channels,
                file_hash=excluded.file_hash
            """, (image_id, fname, path, int(st.st_size), int(w), int(h), int(ch), fhash, self._now_iso(), self._now_iso()))

            # --- color features (32 Bins, 3 dominants) ---
            ca = ColorAnalyzer(k_clusters=3, random_state=42)
            cf = ca.extract_color_features(img, num_bins=32, num_dominant=3)

            cur.execute("""
            INSERT INTO color_features (image_id, dominant_colors, hsv_histogram, bgr_histogram, color_stats)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                dominant_colors=excluded.dominant_colors,
                hsv_histogram=excluded.hsv_histogram,
                bgr_histogram=excluded.bgr_histogram,
                color_stats=excluded.color_stats
            """, (image_id,
                    json.dumps(cf["dominant_colors"]),
                    json.dumps(cf["hsv_histogram"]),
                    json.dumps(cf["bgr_histogram"]),
                    json.dumps(cf["color_stats"])) )

            # --- pHash als "custom_features" ablegen ---
            ph = compute_phash(img)
            custom = {"phash_hex": hex(ph)}
            cur.execute("""
            INSERT INTO advanced_features (image_id, texture_features, deep_embeddings, custom_features)
            VALUES (?, NULL, NULL, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                custom_features=excluded.custom_features
            """, (image_id, json.dumps(custom)))

            self.conn.commit()
            return image_id

        except Exception as e:
            logger.error(f"add_image failed for {path}: {e}")
            self.conn.rollback()
            return None
    
    
    """
    def add_image(self, filepath: str) -> Optional[str]:
        ""
        Fügt ein Bild zur Datenbank hinzu
        
        Args:
            filepath (str): Pfad zur Bilddatei
            
        Returns:
            Optional[str]: Image ID wenn erfolgreich, None sonst
        ""
        try:
            # Image ID generieren
            image_id = self.generate_image_id(filepath)
            
            # Prüfen ob Bild bereits existiert
            if self.image_exists(image_id):
                self.logger.info(f"Bild bereits in DB: {filepath}")
                return image_id
            
            # Metadaten extrahieren
            metadata = self.extract_metadata(filepath)
            if metadata is None:
                return None
            
            # Bild laden für Feature Extraction
            image = cv2.imread(filepath)
            if image is None:
                self.logger.error(f"Kann Bild nicht laden: {filepath}")
                return None
            
            # Features extrahieren
            color_features = self.extract_color_features(image, image_id)
            advanced_features = self.extract_advanced_features(image, image_id)
            
            # In Datenbank speichern
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Image Metadaten speichern
                cursor.execute(""
                    INSERT INTO images (image_id, filename, filepath, file_size, 
                                      width, height, channels, file_hash, 
                                      photographer, created_date, added_to_db)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "", (image_id, metadata['filename'], metadata['filepath'],
                     metadata['file_size'], metadata['width'], metadata['height'],
                     metadata['channels'], metadata['file_hash'], 
                     metadata['photographer'], metadata['created_date'],
                     metadata['added_to_db']))
                
                # Color Features speichern
                if color_features:
                    cursor.execute(""
                        INSERT INTO color_features (image_id, dominant_colors, 
                                                   hsv_histogram, bgr_histogram, color_stats)
                        VALUES (?, ?, ?, ?, ?)
                    "", (image_id, json.dumps(color_features['dominant_colors']),
                         json.dumps(color_features['hsv_histogram']),
                         json.dumps(color_features['bgr_histogram']),
                         json.dumps(color_features['color_stats'])))
                
                # Advanced Features speichern
                if advanced_features:
                    cursor.execute(""
                        INSERT INTO advanced_features (image_id, texture_features, 
                                                     deep_embeddings, custom_features)
                        VALUES (?, ?, ?, ?)
                    "", (image_id, json.dumps(advanced_features['texture_features']),
                         json.dumps(advanced_features['deep_embeddings']),
                         json.dumps(advanced_features['custom_features'])))
                
                conn.commit()
            
            self.logger.info(f"Bild erfolgreich hinzugefügt: {filepath} (ID: {image_id})")
            return image_id
            
        except Exception as e:
            self.logger.error(f"Fehler beim Hinzufügen von Bild {filepath}: {e}")
            return None
    """
    def image_exists(self, image_id: str) -> bool:
        """Prüft ob Bild bereits in DB existiert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM images WHERE image_id = ?", (image_id,))
            return cursor.fetchone() is not None
    
    def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """Holt Metadaten für eine Image ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_color_features(self, image_id: str) -> Optional[Dict]:
        """Holt Color Features für eine Image ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM color_features WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'dominant_colors': json.loads(row[1]),
                    'hsv_histogram': json.loads(row[2]),
                    'bgr_histogram': json.loads(row[3]),
                    'color_stats': json.loads(row[4])
                }
            return None
        
    def get_custom_features(self, image_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT custom_features FROM advanced_features WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if not row or row[0] is None:
                return None
            try:
                return json.loads(row[0])
            except Exception:
                return None

    def get_all_images(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_id, filename, filepath FROM images")
            return [{"image_id": r[0], "filename": r[1], "filepath": r[2]} for r in cursor.fetchall()]
    
    def get_all_image_ids(self) -> List[str]:
        """Holt alle Image IDs aus der Datenbank"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_id FROM images")
            return [row[0] for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict:
        """Gibt Statistiken über die Datenbank zurück"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Anzahl Bilder
            cursor.execute("SELECT COUNT(*) FROM images")
            total_images = cursor.fetchone()[0]
            
            # Anzahl mit Color Features
            cursor.execute("SELECT COUNT(*) FROM color_features")
            color_features_count = cursor.fetchone()[0]
            
            # Anzahl mit Advanced Features
            cursor.execute("SELECT COUNT(*) FROM advanced_features")
            advanced_features_count = cursor.fetchone()[0]
            
            # Datenbankgröße
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_images': total_images,
                'color_features_count': color_features_count,
                'advanced_features_count': advanced_features_count,
                'database_size_mb': db_size / (1024 * 1024)
            }


class ImageLoader:
    """
    Generator für effizientes Laden von Bildern (Projektanforderung)
    Funktioniert mit deinem aktuellen Ordner mit 5 Bildern
    """
    
    def __init__(self, image_database: ImageDatabase):
        """
        Initialisiert den ImageLoader
        
        Args:
            image_database (ImageDatabase): Referenz zur Database
        """
        self.db = image_database
        self.logger = logging.getLogger(__name__)
    
    def scan_directory(self, directory_path: str, extensions: List[str] = None) -> List[str]:
        """
        Scannt Verzeichnis nach Bilddateien
        
        Args:
            directory_path (str): Pfad zum Bildverzeichnis
            extensions (List[str]): Erlaubte Dateierweiterungen
            
        Returns:
            List[str]: Liste der gefundenen Bilddateien
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_files = []
        directory = Path(directory_path)
        
        if not directory.exists():
            self.logger.error(f"Verzeichnis existiert nicht: {directory_path}")
            return image_files
        
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Gefunden: {len(image_files)} Bilder in {directory_path}")
        return [str(f) for f in image_files]
    
    def load_images_generator(self, image_paths: List[str]) -> Generator[Tuple[str, np.ndarray, str], None, None]:
        """
        Generator für das Laden von Bildern
        
        Args:
            image_paths (List[str]): Liste der Bildpfade
            
        Yields:
            Tuple[str, np.ndarray, str]: (image_id, image_array, filepath)
        """
        for filepath in image_paths:
            try:
                # Bild laden
                image = cv2.imread(filepath)
                if image is None:
                    self.logger.warning(f"Kann Bild nicht laden: {filepath}")
                    continue
                
                # Image ID generieren
                image_id = self.db.generate_image_id(filepath)
                
                yield image_id, image, filepath
                
            except Exception as e:
                self.logger.error(f"Fehler beim Laden von {filepath}: {e}")
                continue

# Praktische Hilfsfunktionen
def initialize_database_with_images(image_directory: str, db_path: str = "image_recommender.db") -> ImageDatabase:
    """
    Convenience-Funktion zum Initialisieren der DB mit deinen Bildern
    
    Args:
        image_directory (str): Pfad zu deinem Ordner mit 5 Bildern
        db_path (str): Pfad zur Datenbank
        
    Returns:
        ImageDatabase: Initialisierte Datenbank
    """
    # Database und Loader erstellen
    db = ImageDatabase(db_path)
    loader = ImageLoader(db)
    
    # Bilder hinzufügen
    #stats = loader.bulk_add_directory(image_directory)
    
    print(f"Datenbank initialisiert!")
    print(f"Hinzugefügt: {stats['added']}, Übersprungen: {stats['skipped']}, Fehler: {stats['errors']}")
    print(f"DB Stats: {db.get_database_stats()}")
    
    return db