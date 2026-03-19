import os
import json
import ast
import shutil
import pandas as pd
from pathlib import Path
import kagglehub


# ---------------------------
# CONFIG
# ---------------------------
TOP_K = 2
OUTPUT_FILE = "pairs.json"
MAX_IMAGES = 20000
MAX_ROWS = 20000

PROMPT_TEMPLATE = "a painting that evokes {}"

EMOTIONS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
    "something_else"
]


# ---------------------------
# DOWNLOAD DATASETS
# ---------------------------
def download_datasets():
    wikiart_path = kagglehub.dataset_download("steubk/wikiart")
    artemis_path = kagglehub.dataset_download("samamostafa03/artemis-dataset")

    print("WikiArt path:", wikiart_path)
    print("ArtEmis path:", artemis_path)

    return Path(wikiart_path), Path(artemis_path)


# ---------------------------
# BUILD IMAGE INDEX (LIMITED)
# ---------------------------
def build_image_index(root):
    image_map = {}
    count = 0

    for path in root.rglob("*"):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            key = path.stem.lower()
            image_map[key] = str(path)

            count += 1
            if count >= MAX_IMAGES:
                break

    print(f"Indexed {len(image_map)} images")
    return image_map


# ---------------------------
# LOAD ARTEMIS CSV
# ---------------------------
def load_artemis_csv(artemis_root):
    for p in artemis_root.rglob("image-emotion-histogram.csv"):
        df = pd.read_csv(p)
        print("CSV loaded:", p)
        return df

    raise FileNotFoundError("image-emotion-histogram.csv not found")


# ---------------------------
# PARSE EMOTION HISTOGRAM
# ---------------------------
def get_top_emotions(row, k=2):
    hist = ast.literal_eval(row["emotion_histogram"])

    idx = sorted(range(len(hist)), key=lambda i: hist[i], reverse=True)[:k]

    return [EMOTIONS[i] for i in idx]


# ---------------------------
# NORMALIZE IMAGE ID
# ---------------------------
def normalize_id(name):
    return str(name).lower().replace(" ", "_").replace("-", "_")


# ---------------------------
# MAIN PREPROCESSING
# ---------------------------
def preprocess():
    wikiart_root, artemis_root = download_datasets()

    print("Indexing images...")
    image_map = build_image_index(wikiart_root)

    print("Loading ArtEmis CSV...")
    df = load_artemis_csv(artemis_root)

    # reduce dataset size (important)
    df = df.sample(n=MAX_ROWS, random_state=42)

    ID_COL = "painting"

    pairs = []
    skipped = 0

    print("Creating pairs...")

    for _, row in df.iterrows():
        image_id = normalize_id(row[ID_COL])

        if image_id not in image_map:
            skipped += 1
            continue

        image_path = image_map[image_id]
        emotions = get_top_emotions(row, TOP_K)

        for emotion in emotions:
            pairs.append({
                "image": image_path,
                "text": PROMPT_TEMPLATE.format(emotion)
            })

    print(f"Total pairs: {len(pairs)}")
    print(f"Skipped (no match): {skipped}")

    # save output
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / OUTPUT_FILE, "w") as f:
        json.dump(pairs, f)

    print("Saved to:", output_path / OUTPUT_FILE)

    # CLEAN CACHE (VERY IMPORTANT)
    print("Cleaning dataset cache...")
    shutil.rmtree("/root/.cache/kagglehub", ignore_errors=True)
    print("Cache cleaned")


# ---------------------------
# ENTRY
# ---------------------------
if __name__ == "__main__":
    preprocess()