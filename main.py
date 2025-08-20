import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np

# Eigene Module importieren
from ImageDatabase import ImageDatabase, initialize_database_with_images
from ImageRecommender import ImageRecommender
from SearchImage import SearchEngine


def setup_logging(verbose: bool = False):
    """Setup Logging Configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("image_recommender.log")],
    )


def print_banner():
    """Druckt das Banner"""
    print("=" * 60)
    print("🖼️  IMAGE RECOMMENDER SYSTEM - DEMO")
    print("=" * 60)
    print("Dieses Script testet alle Funktionen des Systems:")
    print("1. Datenbank-Setup mit Bildimport")
    print("2. Feature-Extraktion und Indexierung")
    print("3. Similarity-Suche mit 3 Algorithmen")
    print("4. Performance-Benchmarking")
    print("5. Multi-Input Suche")
    print("=" * 60)


def demo_database_setup(
    image_directory: str, db_path: str = "demo_recommender.db"
) -> ImageDatabase:
    """
    Demonstriert Database Setup und Bildimport

    Args:
        image_directory: Pfad zum Bildverzeichnis
        db_path: Pfad zur Datenbank

    Returns:
        ImageDatabase: Initialisierte Datenbank
    """
    print("\n🗄️  SCHRITT 1: DATENBANK SETUP")
    print("-" * 40)

    # Prüfen ob Verzeichnis existiert
    if not os.path.exists(image_directory):
        print(f"❌ Bildverzeichnis nicht gefunden: {image_directory}")
        print("Erstelle Test-Verzeichnis mit Beispielbildern...")
        create_test_images(image_directory)

    # Datenbank initialisieren
    print(f"📂 Lade Bilder aus: {image_directory}")
    start_time = time.time()

    db = initialize_database_with_images(image_directory, db_path)

    setup_time = time.time() - start_time
    print(f"⏱️  Setup-Zeit: {setup_time:.2f}s")

    # Statistiken anzeigen
    stats = db.get_database_stats()
    print("📊 Datenbank-Statistiken:")
    print(f"   • Bilder gesamt: {stats['total_images']}")
    print(f"   • Color Features: {stats['color_features_count']}")
    print(f"   • Advanced Features: {stats['advanced_features_count']}")
    print(f"   • DB-Größe: {stats['database_size_mb']:.2f} MB")

    return db


def demo_feature_extraction(db: ImageDatabase):
    """
    Demonstriert Feature-Extraktion für ein Beispielbild

    Args:
        db: ImageDatabase Instanz
    """
    print("\n🔬 SCHRITT 2: FEATURE-EXTRAKTION DEMO")
    print("-" * 40)

    # Erstes Bild aus DB holen
    image_ids = db.get_all_image_ids()
    if not image_ids:
        print("❌ Keine Bilder in der Datenbank!")
        return

    image_id = image_ids[0]
    metadata = db.get_image_metadata(image_id)
    color_features = db.get_color_features(image_id)

    print(f"🖼️  Beispielbild: {metadata['filename']}")
    print(f"   • Auflösung: {metadata['width']}x{metadata['height']}")
    print(f"   • Kanäle: {metadata['channels']}")
    print(f"   • Dateigröße: {metadata['file_size'] / 1024:.1f} KB")

    if color_features:
        print("🎨 Color Features:")
        print(f"   • Dominante Farben: {len(color_features['dominant_colors'])} Farben")
        print(f"   • Helligkeit: {color_features['color_stats']['brightness']:.1f}")
        print(f"   • Erste dominante Farbe (BGR): {color_features['dominant_colors'][0]}")


def demo_similarity_search(db: ImageDatabase):
    """
    Demonstriert alle 3 Similarity-Algorithmen

    Args:
        db: ImageDatabase Instanz
    """
    print("\n🔍 SCHRITT 3: SIMILARITY-SUCHE DEMO")
    print("-" * 40)

    # Image Recommender erstellen
    recommender = ImageRecommender(db)

    # Erstes Bild als Query verwenden
    image_ids = db.get_all_image_ids()
    if len(image_ids) < 2:
        print("❌ Brauche mindestens 2 Bilder für Similarity-Suche!")
        return

    query_image_id = image_ids[0]
    query_metadata = db.get_image_metadata(query_image_id)
    query_image_path = query_metadata["filepath"]

    print(f"🎯 Query-Bild: {query_metadata['filename']}")

    # Ähnliche Bilder finden
    start_time = time.time()
    similar_images = recommender.find_similar_images(query_image_path, top_k=3)
    search_time = time.time() - start_time

    print(f"⏱️  Suche dauerte: {search_time:.3f}s")
    print("🏆 Top ähnliche Bilder:")

    for i, result in enumerate(similar_images, 1):
        print(f"   {i}. {result['metadata']['filename']}")
        print(f"      Gesamt-Score: {result['similarity_score']:.3f}")
        print(f"      Color: {result['detailed_scores']['color_similarity']:.3f}")
        print(f"      Embedding: {result['detailed_scores']['embedding_similarity']:.3f}")
        print(f"      Custom: {result['detailed_scores']['custom_similarity']:.3f}")


def demo_multi_input_search(db: ImageDatabase):
    """
    Demonstriert Multi-Input Suche

    Args:
        db: ImageDatabase Instanz
    """
    print("\n🔢 SCHRITT 4: MULTI-INPUT SUCHE DEMO")
    print("-" * 40)

    image_ids = db.get_all_image_ids()
    if len(image_ids) < 3:
        print("❌ Brauche mindestens 3 Bilder für Multi-Input Suche!")
        return

    recommender = ImageRecommender(db)

    # Erste 2 Bilder als Multi-Input verwenden
    input_paths = []
    for i in range(min(2, len(image_ids))):
        metadata = db.get_image_metadata(image_ids[i])
        input_paths.append(metadata["filepath"])
        print(f"🎯 Input {i + 1}: {metadata['filename']}")

    # Multi-Input Suche
    start_time = time.time()
    multi_results = recommender.find_similar_multi_input(
        input_paths, top_k=2, combination_method="average"
    )
    search_time = time.time() - start_time

    print(f"⏱️  Multi-Input Suche dauerte: {search_time:.3f}s")
    print("🏆 Kombinierte Ergebnisse:")

    for i, result in enumerate(multi_results, 1):
        print(f"   {i}. {result['metadata']['filename']}")
        print(f"      Score: {result['combined_similarity_score']:.3f}")


def demo_search_engine(db: ImageDatabase):
    """
    Demonstriert Search Engine mit verschiedenen ANN-Algorithmen

    Args:
        db: ImageDatabase Instanz
    """
    print("\n⚡ SCHRITT 5: SEARCH ENGINE DEMO")
    print("-" * 40)

    algorithms = ["brute_force", "lsh", "tree_based"]

    image_ids = db.get_all_image_ids()
    if not image_ids:
        print("❌ Keine Bilder für Search Engine Demo!")
        return

    # Query-Bild vorbereiten
    query_metadata = db.get_image_metadata(image_ids[0])
    query_path = query_metadata["filepath"]

    print(f"🎯 Query-Bild: {query_metadata['filename']}")
    print("🧪 Teste verschiedene ANN-Algorithmen:")

    for algorithm in algorithms:
        try:
            print(f"\n   📊 {algorithm.upper()}:")

            # Search Engine mit spezifischem Algorithmus
            search_engine = SearchEngine(db, ann_algorithm=algorithm)

            # Suche durchführen
            start_time = time.time()
            results = search_engine.search_similar_images(query_path, k=2)
            search_time = time.time() - start_time

            print(f"      ⏱️  Zeit: {search_time:.4f}s")
            print(f"      📈 Ergebnisse: {len(results)}")

            # Erstes Ergebnis zeigen
            if results:
                best_result = results[0]
                print(
                    f"      🥇 Bestes Match: {best_result['metadata']['filename']} (Score: {best_result['similarity_score']:.3f})"
                )

        except Exception as e:
            print(f"      ❌ Fehler bei {algorithm}: {e}")


def demo_performance_benchmark(db: ImageDatabase):
    """
    Führt Performance-Benchmarks durch

    Args:
        db: ImageDatabase Instanz
    """
    print("\n🚀 SCHRITT 6: PERFORMANCE BENCHMARK")
    print("-" * 40)

    image_ids = db.get_all_image_ids()
    if not image_ids:
        print("❌ Keine Bilder für Benchmark!")
        return

    # Query-Bild vorbereiten
    query_metadata = db.get_image_metadata(image_ids[0])
    query_path = query_metadata["filepath"]

    # Search Engine erstellen
    search_engine = SearchEngine(db, ann_algorithm="brute_force")

    # Benchmark für verschiedene k-Werte
    k_values = [1, 3, 5]
    print(f"🎯 Benchmark Query: {query_metadata['filename']}")
    print(f"📊 Teste verschiedene k-Werte: {k_values}")

    try:
        benchmark_results = search_engine.benchmark_search_performance(query_path, k_values)

        print("\n📈 Benchmark-Ergebnisse:")
        for k, stats in benchmark_results.items():
            print(f"   {k}: {stats['avg_time']:.4f}s ± {stats['std_time']:.4f}s")
            print(f"      Min: {stats['min_time']:.4f}s, Max: {stats['max_time']:.4f}s")
            print(f"      Ergebnisse: {stats['results_count']}")

    except Exception as e:
        print(f"❌ Benchmark-Fehler: {e}")


def create_test_images(directory: str):
    """
    Erstellt Test-Bilder falls keine vorhanden

    Args:
        directory: Verzeichnis für Test-Bilder
    """
    print(f"🎨 Erstelle Test-Bilder in: {directory}")

    os.makedirs(directory, exist_ok=True)

    # Erstelle 5 verschiedene Test-Bilder
    test_images = [
        ("red_image.jpg", (0, 0, 255)),  # Rot
        ("green_image.jpg", (0, 255, 0)),  # Grün
        ("blue_image.jpg", (255, 0, 0)),  # Blau
        ("yellow_image.jpg", (0, 255, 255)),  # Gelb
        ("purple_image.jpg", (255, 0, 255)),  # Magenta
    ]

    for filename, color in test_images:
        # Einfaches farbiges Bild erstellen
        image = np.full((200, 200, 3), color, dtype=np.uint8)

        # Etwas Textur hinzufügen
        noise = np.random.normal(0, 20, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, image)
        print(f"   ✅ {filename} erstellt")


def main():
    """Hauptfunktion für Demo"""

    # Argument parsing
    parser = argparse.ArgumentParser(description="Image Recommender System Demo")
    parser.add_argument(
        "--image_dir",
        "-i",
        default="./test_images",
        help="Pfad zum Bildverzeichnis (default: ./test_images)",
    )
    parser.add_argument(
        "--db_path",
        "-d",
        default="demo_recommender.db",
        help="Pfad zur Datenbank (default: demo_recommender.db)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging aktivieren")
    parser.add_argument(
        "--skip_setup",
        "-s",
        action="store_true",
        help="Database Setup überspringen (falls bereits vorhanden)",
    )

    args = parser.parse_args()

    # Logging setup
    setup_logging(args.verbose)

    # Banner anzeigen
    print_banner()

    try:
        # 1. Database Setup (falls nicht übersprungen)
        if args.skip_setup and os.path.exists(args.db_path):
            print(f"\n⏭️  Überspringe Setup, lade existierende DB: {args.db_path}")
            db = ImageDatabase(args.db_path)
        else:
            db = demo_database_setup(args.image_dir, args.db_path)

        # Prüfen ob Bilder vorhanden
        if len(db.get_all_image_ids()) == 0:
            print("\n❌ Keine Bilder in der Datenbank! Demo kann nicht fortgesetzt werden.")
            return

        # 2. Feature-Extraktion Demo
        demo_feature_extraction(db)

        # 3. Similarity-Suche Demo
        demo_similarity_search(db)

        # 4. Multi-Input Suche Demo
        demo_multi_input_search(db)

        # 5. Search Engine Demo
        demo_search_engine(db)

        # 6. Performance Benchmark
        demo_performance_benchmark(db)

        # Abschluss
        print("\n" + "=" * 60)
        print("🎉 DEMO ERFOLGREICH ABGESCHLOSSEN!")
        print("=" * 60)
        print(f"📁 Datenbank gespeichert in: {args.db_path}")
        print("📝 Logs gespeichert in: image_recommender.log")
        print("\nDas System ist bereit für weitere Tests!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo durch Benutzer abgebrochen.")

    except Exception as e:
        print(f"\n\n❌ Fehler während Demo: {e}")
        logging.exception("Demo-Fehler")
        sys.exit(1)


if __name__ == "__main__":
    main()
