# Bild‑Recommender (schnell & leicht)

Finde für ein Query-Bild die Top‑k ähnlichen Bilder aus großen Sammlungen.

- **Ingest**: TurboJPEG + Threads → HSV/BGR‑Histogramme, pHash in **SQLite (WAL)**
- **Embeddings**: MobileNetV3 → **128D** (BLOB)
- **Suche**: ANN via **hnswlib** (Fallback: NumPy‑Cosine)
- **UI**: Streamlit‑Demo, **Notebook** für UMAP/t‑SNE‑Plots

---

## Voraussetzungen

- Python **3.9–3.12**
- Pakete: siehe `requirements.txt`
- **libjpeg‑turbo** empfohlen (für pyTurboJPEG)

**macOS (Apple Silicon)**
```bash
brew install jpeg-turbo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TURBOJPEG=/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib
```

**Linux (Beispiel)**
```bash
sudo apt-get update && sudo apt-get install -y libjpeg-turbo8
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ggf. TURBOJPEG setzen, falls nötig
```

---

## Schnellstart

### 1) Ingest (Features → SQLite)
Liest Bilder ein, extrahiert Farb-/pHash‑Features und speichert sie in `image_recommender.db`.
```bash
python ImageDatabase.py   --ingest "/pfad/zu/bildern"   --db image_recommender.db   --pool thread --workers 64   --decode-mode auto --jpeg-reduce 8   --bins 16 --k 3 --resize 224   --batch-size 8000 --log-every 1000
```

### 2) Embedding‑Backfill (128D, MPS/GPU)
Berechnet kompakte Deep‑Embeddings (MobileNetV3) und speichert sie als BLOB.
```bash
# One‑shot (ohne Polling)
python EmbedBackfill.py   --db image_recommender.db   --batch 192 --io-workers 16   --decode-mode turbo --jpeg-reduce 8   --log-every 5

# Optional: Polling‑Modus (periodisch neue Bilder verarbeiten)
python EmbedBackfill.py   --db image_recommender.db   --batch 192 --io-workers 16   --decode-mode turbo --jpeg-reduce 8   --log-every 5 --poll-interval 60
```

### 3) Suche (CLI)
Top‑k ähnliche Bilder zu einem Query.
```bash
python cli.py query   --db image_recommender.db   --image "/pfad/query.jpg" --topk 5   --ann-threshold 1000
```

**Multi‑Query**
```bash
python cli.py query-multi   --db image_recommender.db   --images /p1.jpg /p2.jpg   --topk 5
```

### 4) Streamlit‑UI
```bash
streamlit run streamlit_app.py
```
Trage in der Sidebar den DB‑Pfad ein, lade ein Bild hoch, passe Gewichte an und starte die Suche.

---

- Autoren: *[Julia Moor & Dalia Salih]*