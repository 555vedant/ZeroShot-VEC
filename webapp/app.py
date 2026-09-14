import os
import sys
import json
import torch
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

# Add project root to sys path so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import format_emotion_prompt, normalize_emotion_text, resolve_image_path
from src.inference import SearchEngine
from src.model import CLIPFineTuner
from transformers import CLIPProcessor
from utils.helpers import load_json
from utils.config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define secure_filename safely fallback
try:
    from werkzeug.utils import secure_filename
except ImportError:
    import re
    def secure_filename(filename):
        return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

print("Loading Model...", flush=True)

webapp_dir = Path(__file__).resolve().parent
local_checkpoint = webapp_dir / "clip_model.pth"
config_checkpoint = Path(Config.CHECKPOINT_FILE)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = None
if local_checkpoint.exists():
    checkpoint_path = local_checkpoint
elif config_checkpoint.exists():
    checkpoint_path = config_checkpoint

# 1. Load the shared model and artwork index.
search_engine = None
if checkpoint_path is not None:
    try:
        search_engine = SearchEngine(
            checkpoint_path=checkpoint_path,
            image_dir=app.config['UPLOAD_FOLDER'],
        )
        model = search_engine.model
        processor = search_engine.processor
    except ValueError as exc:
        print(f"Warning: {exc}")
        print("Starting without the upload image index. Upload an image before searching.")
        model = CLIPFineTuner().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_checkpoint_state_dict(state)
        processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=False)
        search_engine = None
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    model = CLIPFineTuner().to(device)
    print("Warning: No checkpoint found. Using base model.")

model.eval()
print("Model and Processor loaded successfully!", flush=True)

EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "awe",
    "serenity",
    "loneliness",
    "melancholy",
]

LOW_COSINE_THRESHOLD = 0.045
MID_COSINE_THRESHOLD = 0.09
HIGH_COSINE_THRESHOLD = 0.2


def _load_rejection_thresholds():
    path = Path(Config.REJECTION_CALIBRATION_FILE)
    if not path.exists():
        return None
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if "top1_threshold" not in values or "margin_threshold" not in values:
            return None
        return values
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


REJECTION_THRESHOLDS = _load_rejection_thresholds()


def _select_emotion(emotion_scores, thresholds):
    rejection = "Not a valid painting/emotion image"
    if not emotion_scores or thresholds is None:
        return rejection, True, 0.0

    top_result = emotion_scores[0]
    second_score = emotion_scores[1]["score"] if len(emotion_scores) > 1 else 0.0
    margin = top_result["score"] - second_score
    rejected = (
        top_result["score"] < thresholds["top1_threshold"]
        or margin < thresholds["margin_threshold"]
    )
    return (rejection if rejected else top_result["emotion"], rejected, margin)

try:
    raw_records = load_json(Config.DATA_FILE)
    dataset_labels = sorted(
        {
            normalize_emotion_text(record.get("text", ""))
            for record in raw_records
            if record.get("text")
        }
    )
except (OSError, TypeError, ValueError):
    dataset_labels = []

EMOTION_LABELS = sorted(set(EMOTION_LABELS).union(dataset_labels))


def _calibrate_probability(prob_percent, cosine_sim):
    scale = 1.0
    if cosine_sim < LOW_COSINE_THRESHOLD:
        ratio = max(0.0, cosine_sim / LOW_COSINE_THRESHOLD)
        scale = max(0.1, 0.4 * ratio)
    elif cosine_sim < MID_COSINE_THRESHOLD:
        ratio = (cosine_sim - LOW_COSINE_THRESHOLD) / (MID_COSINE_THRESHOLD - LOW_COSINE_THRESHOLD)
        scale = 0.4 + 0.6 * max(0.0, min(1.0, ratio))
    elif cosine_sim >= HIGH_COSINE_THRESHOLD:
        scale = 1.05

    calibrated = prob_percent * scale
    return max(0.0, min(99.9, calibrated))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/dataset-image')
def dataset_image():
    from flask import send_file
    raw_path = request.args.get("path", "")
    resolved = resolve_image_path(raw_path)
    if resolved is None:
        return "Not found", 404
    return send_file(str(resolved))


@app.route('/search', methods=['POST'])
def search_images():
    query = request.form.get('query', '').strip()
    try:
        top_k = max(1, min(int(request.form.get('top_k', Config.SEARCH_TOP_K)), 50))
    except (TypeError, ValueError):
        top_k = Config.SEARCH_TOP_K

    if not query:
        return render_template(
            'index.html',
            error='Enter a text or emotion query to search.',
            labels=EMOTION_LABELS,
        )
    if search_engine is None or not search_engine.data:
        return render_template(
            'index.html',
            error='No uploaded images are available for text-to-image retrieval. Upload images first.',
            labels=EMOTION_LABELS,
        )

    try:
        search_results = search_engine.search_with_scores(query, top_k=top_k)
    except Exception as exc:
        return render_template(
            'index.html',
            error=f'Error during image retrieval: {exc}',
            labels=EMOTION_LABELS,
        )

    return render_template(
        'index.html',
        labels=EMOTION_LABELS,
        search_query=query,
        search_results=search_results,
        search_top_k=top_k,
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    global search_engine, model, processor

    if request.method == 'POST':
        selected_label = request.form.get('label', '').strip().lower()
        if selected_label not in EMOTION_LABELS:
            return render_template('index.html', error='Invalid label selected.', labels=EMOTION_LABELS)

        if 'image' not in request.files:
            return render_template('index.html', error='No image uploaded.')
            
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file.')
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if search_engine is None and checkpoint_path is not None:
                try:
                    search_engine = SearchEngine(
                        checkpoint_path=checkpoint_path,
                        image_dir=app.config['UPLOAD_FOLDER'],
                    )
                    model = search_engine.model
                    processor = search_engine.processor
                except ValueError:
                    search_engine = None
            
            try:
                # Open image
                image = Image.open(filepath).convert("RGB")
            except Exception as e:
                return render_template('index.html', error=f"Invalid image: {str(e)}")
            
            error = None
            score = None
            emotion_scores = []
            predicted_label = "Not a valid painting/emotion image"
            model_prompt = None
            try:
                if search_engine is None:
                    raise RuntimeError("Upload an image before running emotion inference.")

                emotion_scores = search_engine.score_image_against_emotions(image, EMOTION_LABELS)
                top_result = emotion_scores[0]
                predicted_label, is_rejected, margin = _select_emotion(
                    emotion_scores, REJECTION_THRESHOLDS
                )
                if REJECTION_THRESHOLDS is None:
                    error = "Validation thresholds unavailable; run src/evaluate.py to calibrate rejection."
                if not is_rejected:
                    model_prompt = format_emotion_prompt(predicted_label)
                score = f"{top_result['score']:.4f}"

            except Exception as e:
                error = f"Error during inference: {str(e)}"
            
            return render_template(
                'index.html', 
                prompt=predicted_label,
                model_prompt=model_prompt,
                image_path=filename, 
                score=score,
                emotion_scores=emotion_scores,
                error=error,
                labels=EMOTION_LABELS,
                selected_label=selected_label
            )
            
    return render_template('index.html', labels=EMOTION_LABELS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
