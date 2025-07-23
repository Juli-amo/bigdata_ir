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
    
    def color_histogram_similarity(self, image1, image2, method=cv2.HISTCMP_CORREL, 
                                 color_space='HSV', show_plots=False):
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
    
    def get_dominant_colors(self, image):
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
    
    def _calculate_normalized_histogram(self, image, channel):
        """Berechnet normalisiertes Histogramm für einen Kanal"""
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _plot_histogram(self, hist1, hist2, channel_name, subplot_index):
        """Plottet Histogramm für einen Kanal"""
        plt.subplot(2, 3, subplot_index + 1)
        plt.plot(hist1, label=f'{channel_name} Bild 1', alpha=0.7)
        plt.plot(hist2, label=f'{channel_name} Bild 2', alpha=0.7)
        plt.title(f'Kanal: {channel_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)


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
    
    def extract_texture_features(self, image):
        """
        Extrahiert Textur-Features aus einem Bild
        
        Args:
            image (np.array): Input-Bild
            
        Returns:
            dict: Dictionary mit Textur-Features
        """
        # Konvertierung zu Graustufen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Beispiel-Features (können erweitert werden)
        features = {
            'mean': np.mean(gray),
            'std': np.std(gray),
            'variance': np.var(gray)
        }
        
        return features
    
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
