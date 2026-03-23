import json
import ast
import shutil
import pandas as pd
from pathlib import Path
import kagglehub
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import Config


# CONFIG
TOP_K = 2
MAX_IMAGES = 40000   
MAX_ROWS = 40000
CLEAN_KAGGLE_CACHE = False

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
    "something_else",
]


def _find_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct

    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find '{name}' under {root}")
    return matches[0]


def get_dataset_roots():
    wikiart_root = Config.BASE_PATH
    artemis_root = Config.ARTEMIS_PATH

    if wikiart_root.exists() and artemis_root.exists():
        print("Using dataset paths from Config")
        print("WikiArt path:", wikiart_root)
        print("ArtEmis path:", artemis_root)
        return wikiart_root, artemis_root

    print("Config paths not found. Downloading via kagglehub...")
    wikiart_path = Path(kagglehub.dataset_download("steubk/wikiart"))
    artemis_path = Path(kagglehub.dataset_download("samamostafa03/artemis-dataset"))

    print("WikiArt path:", wikiart_path)
    print("ArtEmis path:", artemis_path)
    return wikiart_path, artemis_path


# IMAGE INDEX (USING classes.csv)
def build_image_index(wikiart_root):
    csv_path = _find_file(wikiart_root, "classes.csv")
    data_root = csv_path.parent

    df = pd.read_csv(csv_path)

    image_map = {}

    for i, row in df.iterrows():
        if i >= MAX_IMAGES:
            break

        file_path = str(row["filename"]).replace("\\", "/")
        full_path = data_root / file_path

        if not full_path.exists():
            continue

        key = Path(file_path).stem.lower()
        image_map[key] = file_path

    print(f"Indexed {len(image_map)} images")
    return image_map


# LOAD ARTEMIS CSV
def load_artemis_csv(artemis_root):
    csv_path = _find_file(artemis_root, "image-emotion-histogram.csv")
    df = pd.read_csv(csv_path)
    print("CSV loaded:", csv_path)
    return df


# PARSE EMOTIONS
def get_top_emotions(row, k=2):
    hist = ast.literal_eval(row["emotion_histogram"])

    idx = sorted(range(len(hist)), key=lambda i: hist[i], reverse=True)[:k]

    return [EMOTIONS[i] for i in idx]


# PREPROCESSING
def preprocess():
    wikiart_root, artemis_root = get_dataset_roots()

    print("Building image index using classes.csv...")
    image_map = build_image_index(wikiart_root)

    print("Loading ArtEmis CSV...")
    df = load_artemis_csv(artemis_root)

    # Reduce dataset size but do not request more rows than available.
    sample_n = min(MAX_ROWS, len(df))
    df = df.sample(n=sample_n, random_state=42)

    ID_COL = "painting"

    pairs = []
    skipped = 0
    matched = 0

    print("Creating pairs...")

    for _, row in df.iterrows():
        image_id = str(row[ID_COL]).lower()

        if image_id not in image_map:
            skipped += 1
            continue

        matched += 1
        image_path = image_map[image_id]

        emotions = get_top_emotions(row, TOP_K)

        for emotion in emotions:
            pairs.append({
                "image": image_path,
                "text": PROMPT_TEMPLATE.format(emotion)
            })

    print(f"Matched: {matched}")
    print(f"Skipped: {skipped}")
    print(f"Total pairs: {len(pairs)}")

    Config.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f)

    print("Saved to:", Config.DATA_FILE)

    if CLEAN_KAGGLE_CACHE:
        cache_path = Path.home() / ".cache" / "kagglehub"
        print(f"Cleaning dataset cache at: {cache_path}")
        shutil.rmtree(cache_path, ignore_errors=True)
        print("Cache cleaned")


if __name__ == "__main__":
    preprocess()