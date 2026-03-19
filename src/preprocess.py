import os
import json
import pandas as pd
from pathlib import Path
import kagglehub


# ---------------------------
# CONFIG
# ---------------------------
TOP_K = 2
OUTPUT_FILE = "pairs.json"

PROMPT_TEMPLATE = "a painting that evokes {}"


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
# FIND ALL IMAGES
# ---------------------------
def build_image_index(root):
    image_map = {}

    for path in root.rglob("*"):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_id = path.stem.lower()
            image_map[image_id] = str(path)

    return image_map


# ---------------------------
# LOAD ARTEMIS CSV
# ---------------------------
def load_artemis_csv(artemis_root):
    # find histogram csv
    csv_path = None
    for p in artemis_root.rglob("image-emotion-histogram.csv"):
        csv_path = p
        break

    if csv_path is None:
        raise FileNotFoundError("image-emotion-histogram.csv not found")

    df = pd.read_csv(csv_path)
    return df


# ---------------------------
# EXTRACT TOP-K EMOTIONS
# ---------------------------
def get_top_emotions(row, emotion_cols, k=2):
    values = row[emotion_cols].values
    indices = values.argsort()[::-1][:k]

    return [emotion_cols[i] for i in indices]


# ---------------------------
# MAIN PREPROCESSING
# ---------------------------
def preprocess():
    wikiart_root, artemis_root = download_datasets()

    print("Indexing images...")
    image_map = build_image_index(wikiart_root)

    print("Loading ArtEmis CSV...")
    df = load_artemis_csv(artemis_root)

    # emotion columns (all except id/image column)
    emotion_cols = [col for col in df.columns if col not in ["id"]]

    pairs = []
    skipped = 0

    print("Creating pairs...")

    for _, row in df.iterrows():
        image_id = str(row["id"]).lower()

        if image_id not in image_map:
            skipped += 1
            continue

        image_path = image_map[image_id]

        emotions = get_top_emotions(row, emotion_cols, TOP_K)

        for emotion in emotions:
            text = PROMPT_TEMPLATE.format(emotion)

            pairs.append({
                "image": image_path,
                "text": text
            })

    print(f"Total pairs: {len(pairs)}")
    print(f"Skipped (no image match): {skipped}")

    # save
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / OUTPUT_FILE, "w") as f:
        json.dump(pairs, f)

    print("Saved to:", output_path / OUTPUT_FILE)


# ---------------------------
# ENTRY
# ---------------------------
if __name__ == "__main__":
    preprocess()