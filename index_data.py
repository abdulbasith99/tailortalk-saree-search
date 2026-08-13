"""
TailorTalk - Backend Indexing Script
=====================================
Scans the 'data/' folder for saree images, extracts visual embeddings
using a CLIP model, and stores them in a local ChromaDB collection.

IMPORTANT: This script uses the SAME multi-view embedding fusion as
app.py's query-time embedding extraction. Both sides (indexing and
querying) must use identical embedding logic, otherwise search
results will be inaccurate — the query vector and the stored vectors
must live in the same "space" to be meaningfully comparable.

Usage:
    python index_data.py
"""

import csv
import sys
from pathlib import Path

import chromadb
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "saree_images"
METADATA_FILE = Path("image_metadata.csv")
EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-16"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the CLIP embedding model from sentence-transformers."""
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("[INFO] Model loaded successfully.")
    return model


def load_metadata() -> dict:
    """Load the image_metadata.csv mapping filenames to product info."""
    metadata_map = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("filename"):
                    metadata_map[row["filename"]] = {
                        "name": row.get("name", ""),
                        "sku": row.get("sku", ""),
                        "url": row.get("url", ""),
                    }
        print(f"[INFO] Loaded metadata for {len(metadata_map)} images from {METADATA_FILE}")
    else:
        print(f"[WARNING] {METADATA_FILE} not found — using filename-only metadata.")
    return metadata_map


def get_image_files(data_dir: Path) -> list[Path]:
    """Scan the data directory for saree image files."""
    if not data_dir.exists():
        print(f"[ERROR] Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    image_files = [
        f for f in data_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    ]

    if not image_files:
        print(f"[WARNING] No image files found in '{data_dir}'.")
    else:
        print(f"[INFO] Found {len(image_files)} image file(s) in '{data_dir}'.")

    return sorted(image_files)


def extract_multi_view_embedding(model: SentenceTransformer, image: Image.Image):
    """
    Extract a richer embedding by fusing multiple views of the saree image:
    full, center-crop, top-border, and bottom-border. This captures
    border/pallu work, motifs, fabric, and overall colour composition.

    NOTE: This must stay identical to _extract_multi_view_embedding() in
    app.py, since query-time embeddings are compared directly against
    the embeddings stored here.
    """
    img = image.convert("RGB")
    width, height = img.size

    center_crop = img.crop((int(width * 0.15), int(height * 0.2),
                            int(width * 0.85), int(height * 0.8)))
    top_strip = img.crop((0, 0, width, int(height * 0.35)))
    bottom_strip = img.crop((0, int(height * 0.65), width, height))

    views = [img, center_crop, top_strip, bottom_strip]
    embeddings = [model.encode(v.convert("RGB"), convert_to_numpy=True) for v in views]

    # Normalise each view embedding, then average for fusion
    normed = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
    combined = sum(normed) / len(normed)
    combined = combined / (np.linalg.norm(combined) + 1e-8)
    return combined


def extract_embedding(model: SentenceTransformer, image_path: Path):
    """Extract fused multi-view embedding from an image file."""
    img = Image.open(image_path).convert("RGB")
    combined = extract_multi_view_embedding(model, img)
    return combined.tolist()


def index_images(
    model: SentenceTransformer,
    image_files: list[Path],
    collection: chromadb.Collection,
    metadata_map: dict,
) -> None:
    """Generate embeddings for each image and store them in ChromaDB."""
    if not image_files:
        print("[INFO] No images to index. Exiting.")
        return

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for idx, img_path in enumerate(image_files, start=1):
        try:
            print(f"[INFO] ({idx}/{len(image_files)}) Processing: {img_path.name}")
            embedding = extract_embedding(model, img_path)

            # Get metadata from the CSV mapping if available
            meta = metadata_map.get(img_path.name, {})
            product_name = meta.get("name", "")
            sku = meta.get("sku", "")
            url = meta.get("url", "")

            # Fallback: derive name from filename
            if not product_name:
                stem = img_path.stem
                parts = stem.split("_", 1)
                if len(parts) > 1:
                    product_name = parts[1]
                else:
                    product_name = stem

            ids.append(img_path.stem)
            embeddings.append(embedding)
            metadatas.append({
                "filename": img_path.name,
                "filepath": str(img_path),
                "file_size_bytes": str(img_path.stat().st_size),
                "name": product_name,
                "sku": sku,
                "url": url,
            })
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")

    if ids:
        # Batch add all at once (ChromaDB handles this well)
        print(f"[INFO] Adding {len(ids)} images to ChromaDB collection...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[INFO] Indexed {len(ids)} image(s) into collection '{COLLECTION_NAME}'.")
    else:
        print("[WARNING] No images were successfully indexed.")


def main() -> None:
    """Main entry point for the indexing script."""
    print("=" * 60)
    print("TailorTalk - Image Indexing (multi-view embeddings)")
    print("=" * 60)

    # Load the embedding model
    model = load_embedding_model()

    # Load metadata map
    metadata_map = load_metadata()

    # Initialize a local (persistent) ChromaDB client
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Always start fresh: delete any existing collection so old
    # single-view embeddings don't mix with new multi-view embeddings.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"[INFO] Deleted existing collection: '{COLLECTION_NAME}' (will rebuild fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[INFO] Created new collection: '{COLLECTION_NAME}' (cosine distance)")

    # Scan for images
    image_files = get_image_files(DATA_DIR)

    # Index all images
    index_images(model, image_files, collection, metadata_map)

    # Summary
    count = collection.count()
    print(f"[INFO] Total images in collection: {count}")
    print("=" * 60)
    print("Indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()