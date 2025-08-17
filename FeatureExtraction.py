import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ColorAnalyzer:
    """
    Klasse für farbbasierte Bildanalyse und Ähnlichkeitsberechnung
    """
    
    def __init__(self, k_clusters=3, random_state=42):
        """
        Initialisiert den ColorAnalyzer
        
        Args:
            k_clusters (int): Anzahl der dominanten Farben für K-Means
            random_state (int): Seed für reproduzierbare Ergebnisse
        """
        self.k_clusters = k_clusters
        self.random_state = random_state
    
    def color_histogram_similarity(self, image1, image2, color_space='HSV', num_bins=32,
                               use_opencv=False,
                               show_plots=False):
        """
        Berechnet die Ähnlichkeit zwischen zwei Bildern basierend auf Farbhistogrammen
        
        Args:
            image1 (np.array): Erstes Bild
            image2 (np.array): Zweites Bild
            method (int): Methode für Histogrammvergleich
            color_space (str): Farbraum ('HSV' oder 'BGR')
            show_plots (bool): Ob Histogramme angezeigt werden sollen
            
        Returns:
            float: Ähnlichkeitswert zwischen 0 und 1
        """
        # Farbraumkonvertierung
        img1, img2 = self._convert_color_space(image1, image2, color_space)
        channels = self._get_channel_names(color_space)
        # Kanal-Ranges bestimmen
        if color_space.upper() == 'HSV':
            ranges = [180, 256, 256]  # H,S,V
        else:  # BGR
            ranges = [256, 256, 256]

        hists1, hists2, sims = [], [], []

        if show_plots:
            plt.figure(figsize=(12, 6))

        for i, ch_name in enumerate(channels):
            h1 = self._calculate_normalized_histogram(img1, i, num_bins=num_bins, range_max=ranges[i])
            h2 = self._calculate_normalized_histogram(img2, i, num_bins=num_bins, range_max=ranges[i])
            hists1.append(h1); hists2.append(h2)

            if use_opencv:
                # OpenCV Korrelation ([-1,1]) -> auf [0,1] abbilden
                sim_raw = cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_CORREL)
                sim = max(0.0, min(1.0, (sim_raw + 1.0) * 0.5))
            else:
                # Cosine-Similarity
                num = float(np.dot(h1, h2))
                den = float(np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
                sim = max(0.0, min(1.0, num / den))
            sims.append(sim)

            if show_plots:
                plt.subplot(2, 3, i + 1)
                plt.plot(h1, label=f'{ch_name} Bild 1', alpha=0.7)
                plt.plot(h2, label=f'{ch_name} Bild 2', alpha=0.7)
                plt.title(f'Kanal: {ch_name}')
                plt.legend(); plt.grid(True, alpha=0.3)

        if show_plots:
            lt.tight_layout(); plt.show()

        return float(np.mean(sims))

        """
        similarities = []
        
        if show_plots:
            plt.figure(figsize=(12, 6))
        
        for i, channel_name in enumerate(channels):
            # Histogramme berechnen und normalisieren
            hist1 = self._calculate_normalized_histogram(img1, i)
            hist2 = self._calculate_normalized_histogram(img2, i)
            
            # Visualisierung
            if show_plots:
                self._plot_histogram(hist1, hist2, channel_name, i)
            
            # Ähnlichkeit berechnen
            similarity = cv2.compareHist(hist1, hist2, method)
            similarities.append(similarity)
        
        if show_plots:
            plt.tight_layout()
            plt.show()
        
        # Gesamtähnlichkeit als Mittelwert
        overall_similarity = np.mean(similarities)
        return max(0.0, min(1.0, overall_similarity))
        """
    
    def get_dominant_colors(self, image, max_samples: int = 10000):
        """
        Dominante Farben via k-means (auf Lab für bessere Wahrnehmungsnähe),
        Sampling für Speed, Rückgabe als BGR-Farben (k x 3, dtype float32).
        """
        if image is None:
            raise ValueError("get_dominant_colors: image is None")

        # BGR -> Lab (k-means in Lab ist stabiler bzgl. Wahrnehmung)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype("float32")

        # Sampling
        n = lab.shape[0]
        if n > max_samples:
            idx = np.random.RandomState(self.random_state).choice(n, size=max_samples, replace=False)
            lab_sample = lab[idx]
        else:
            lab_sample = lab

        K = int(self.k_clusters)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _compactness, _labels, centers_lab = cv2.kmeans(
            lab_sample, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )  # (K,3) in Lab

        # Lab -> BGR zum Anzeigen/Weiterverwenden
        centers_lab_u8 = np.clip(centers_lab, 0, 255).astype("uint8").reshape(-1, 1, 3)
        centers_bgr = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2BGR).reshape(K, 3).astype("float32")

        # nach Helligkeit sortieren (Summe der Kanäle als einfache Helligkeit)
        order = np.argsort(centers_bgr.sum(axis=1))
        return centers_bgr[order]




    def get_dominant_colorsv2(self, image):
        """
        Extrahiert dominante Farben aus einem Bild mit K-Means
        
        Args:
            image (np.array): Input-Bild im BGR-Format
            
        Returns:
            np.array: Array der dominanten Farben (k×3)
        """
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.k_clusters, n_init=10, 
                       random_state=self.random_state)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        # Sortierung nach Helligkeit
        brightness = np.sum(colors, axis=1)
        order = np.argsort(brightness)
        
        return colors[order]
    
    def calculate_color_distance(self, colors1, colors2):
        """
        Berechnet die Distanz zwischen zwei Farbpaletten
        
        Args:
            colors1 (np.array): Erste Farbpalette
            colors2 (np.array): Zweite Farbpalette
            
        Returns:
            float: Euklidische Distanz zwischen den Farbpaletten
        """
        return np.linalg.norm(colors1 - colors2)
    
    def analyze_dominant_colors(self, image1, image2, show_palettes=False):
        """
        Komplette Analyse der dominanten Farben zweier Bilder
        
        Args:
            image1 (np.array): Erstes Bild
            image2 (np.array): Zweites Bild
            show_palettes (bool): Ob Farbpaletten angezeigt werden sollen
            
        Returns:
            dict: Dictionary mit Farbpaletten und Distanz
        """
        colors1 = self.get_dominant_colors(image1)
        colors2 = self.get_dominant_colors(image2)
        
        if show_palettes:
            self.plot_color_palette(colors1, "Dominante Farben Bild 1")
            self.plot_color_palette(colors2, "Dominante Farben Bild 2")
        
        distance = self.calculate_color_distance(colors1, colors2)
        
        return {
            'colors1': colors1,
            'colors2': colors2,
            'distance': distance
        }
    
    def plot_color_palette(self, colors, title):
        """
        Visualisiert eine Farbpalette
        
        Args:
            colors (np.array): Array der Farben (k×3)
            title (str): Titel der Visualisierung
        """
        plt.figure(figsize=(6, 2))
        for i, color in enumerate(colors):
            # Konvertierung zu RGB und Normalisierung auf [0,1]
            rgb = color[::-1] / 255.0
            plt.fill_between([i, i+1], 0, 1, color=rgb)
        
        plt.title(title)
        plt.axis('off')
        plt.xlim(0, len(colors))
        plt.ylim(0, 1)
        plt.show()
    
    def _convert_color_space(self, image1, image2, color_space):
        """Konvertiert Bilder in den gewünschten Farbraum"""
        if color_space.upper() == 'HSV':
            img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
            img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
        elif color_space.upper() == 'BGR':
            img1, img2 = image1, image2
        else:
            raise ValueError("color_space muss 'HSV' oder 'BGR' sein.")
        
        return img1, img2
    
    def _get_channel_names(self, color_space):
        """Gibt die Kanalnamen für den Farbraum zurück"""
        return ['H', 'S', 'V'] if color_space.upper() == 'HSV' else ['B', 'G', 'R']
    
    def _calculate_normalized_histogram(self, image, channel, num_bins=32, range_max=256):
        """Berechnet L1-normalisiertes Histogramm für einen Kanal mit frei wählbarer Binzahl/Range."""
        hist = cv2.calcHist([image], [channel], None, [num_bins], [0, range_max]).astype("float32")
        s = float(hist.sum()) + 1e-8
        return (hist / s).flatten()
    
    def _plot_histogram(self, hist1, hist2, channel_name, subplot_index):
        """Plottet Histogramm für einen Kanal"""
        plt.subplot(2, 3, subplot_index + 1)
        plt.plot(hist1, label=f'{channel_name} Bild 1', alpha=0.7)
        plt.plot(hist2, label=f'{channel_name} Bild 2', alpha=0.7)
        plt.title(f'Kanal: {channel_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def extract_color_features(self, image, num_bins: int = 32, num_dominant: int = None):
        """
        Returns dict with:
          - dominant_colors: list[[B,G,R], ...]
          - hsv_histogram: [H_bins, S_bins, V_bins] (each normalized)
            - bgr_histogram: [B_bins, G_bins, R_bins] (each normalized)
            - color_stats: {"brightness": mean(V), "mean_bgr": [...], "std_bgr":[...]}
        """

        if image is None:
            raise ValueError("image is None")

        # ensure 3-channel BGR
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image

        def _norm(h):
            s = float(h.sum()) + 1e-8
            return (h / s).flatten().tolist()

        # BGR hists
        b = cv2.calcHist([bgr],[0],None,[num_bins],[0,256]).astype("float32")
        g = cv2.calcHist([bgr],[1],None,[num_bins],[0,256]).astype("float32")
        r = cv2.calcHist([bgr],[2],None,[num_bins],[0,256]).astype("float32")
        bgr_hist = [_norm(b), _norm(g), _norm(r)]

        # HSV hists
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv],[0],None,[num_bins],[0,180]).astype("float32")
        s = cv2.calcHist([hsv],[1],None,[num_bins],[0,256]).astype("float32")
        v = cv2.calcHist([hsv],[2],None,[num_bins],[0,256]).astype("float32")
        hsv_hist = [_norm(h), _norm(s), _norm(v)]

        # stats
        brightness = float(hsv[:,:,2].mean())
        mean_bgr = [float(bgr[:,:,c].mean()) for c in range(3)]
        std_bgr  = [float(bgr[:,:,c].std()) for c in range(3)]
        stats = {"brightness": brightness, "mean_bgr": mean_bgr, "std_bgr": std_bgr}

        # k-means dominant colors
        px = bgr.reshape(-1,3).astype("float32")
        max_samples= 10000
        if px.shape[0] > max_samples:
            step=int(np.ceil(px.shape[0]/max_samples)); px=px[::step]
        K=max(1,int(num_dominant))
        crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
        _comp,_lab,centers = cv2.kmeans(px,K,None,crit,5,cv2.KMEANS_PP_CENTERS)
        centers = centers.clip(0,255).astype("int32").tolist()

        return {
            "dominant_colors": centers,
            "hsv_histogram": hsv_hist,
            "bgr_histogram": bgr_hist,
            "color_stats": stats,
        }


class ImageFeatureExtractor:
    """
    Klasse für erweiterte Feature-Extraktion und Bildverarbeitung
    """
    
    def __init__(self):
        """Initialisiert den Feature-Extractor"""
        pass
    
    def image_to_vector(self, images):
        """
        Konvertiert Bilder in Vektoren
        
        Args:
            images (list): Liste von Bildern
            
        Returns:
            list: Liste von Bildvektoren
        """
        vector_lists = []
        
        for image in images:
            vector = image.reshape(-1, 3)
            vector_list = [vector[i] for i in range(vector.shape[0])]
            vector_lists.append(vector_list)
        
        return vector_lists
    
    
    def extract_texture_features(self, image, num_bins: int = 32):
        """
        Einfache, aber nützliche Texturfeatures:
        - Gradienten-Magnitude-Histogramm (Sobel)
        - Laplacian-Varianz (Kanten/Fokus)
        - Entropie (Grauwertverteilung)
        + klassische mean/std/var (Kompatibilität)
        """
        if image is None:
            raise ValueError("image is None")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        t_hist = cv2.calcHist([mag], [0], None, [num_bins], [0, 255]).astype("float32")
        t_hist = (t_hist / (t_hist.sum() + 1e-8)).flatten().tolist()

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(lap.var())

        h = cv2.calcHist([gray], [0], None, [256], [0, 256]).astype("float32").flatten()
        p = h / (h.sum() + 1e-8)
        entropy = float(-np.sum(np.where(p > 0, p * np.log2(p), 0.0)))

        return {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'variance': float(np.var(gray)),
            'texture_hist': t_hist,
            'lap_var': lap_var,
            'entropy': entropy
        }
    
    def calculate_similarity_score(self, image1, image2, method='histogram', **kwargs):
        """
        Berechnet Ähnlichkeits-Score zwischen zwei Bildern
        
        Args:
            image1 (np.array): Erstes Bild
            image2 (np.array): Zweites Bild
            method (str): Methode für Ähnlichkeitsberechnung
            **kwargs: Zusätzliche Parameter
            
        Returns:
            float: Ähnlichkeits-Score
        """
        if method == 'histogram':
            color_analyzer = ColorAnalyzer()
            return color_analyzer.color_histogram_similarity(image1, image2, **kwargs)
        elif method == 'dominant_colors':
            color_analyzer = ColorAnalyzer()
            result = color_analyzer.analyze_dominant_colors(image1, image2)
            # Normalisierung der Distanz zu einem Ähnlichkeits-Score
            max_distance = 442  # Maximale mögliche Distanz im RGB-Raum
            similarity = 1 - (result['distance'] / max_distance)
            return max(0.0, similarity)
        else:
            raise ValueError(f"Unbekannte Methode: {method}")


def compute_phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    Perceptual hash (64-bit): to gray -> resize -> DCT -> take top-left block -> threshold by median -> pack bits.
    Returns integer hash (nutze hex() für Speicherung als String).
    """
    if image is None:
        raise ValueError("compute_phash: image is None")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    size = hash_size * highfreq_factor
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR).astype("float32")
    dct = cv2.dct(gray)
    low = dct[:hash_size, :hash_size]
    med = float(np.median(low))
    bits = (low > med).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)