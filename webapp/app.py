import os
import sys
import torch
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

# Add project root to sys path so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import CLIPFineTuner
from src.dataset import format_emotion_prompt, resolve_image_path
from transformers import CLIPProcessor
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

# 1. Load Model
model = CLIPFineTuner().to(device)
checkpoint_path = None
if local_checkpoint.exists():
    checkpoint_path = local_checkpoint
elif config_checkpoint.exists():
    checkpoint_path = config_checkpoint

if checkpoint_path is not None:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_checkpoint_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print("Warning: No checkpoint found. Using base model.")
    
model.eval()

# 2. Load Processor
processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=False)
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

@app.route('/', methods=['GET', 'POST'])
def index():
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
            
            try:
                # Open image
                image = Image.open(filepath).convert("RGB")
            except Exception as e:
                return render_template('index.html', error=f"Invalid image: {str(e)}")
            
            error = None
            model_prompt = format_emotion_prompt(selected_label)
            try:
                # Process inputs (match training prompt format)
                inputs = processor(
                    text=[model_prompt],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                
                with torch.no_grad():
                    # Compute logits the same way as training/eval (temperature-scaled)
                    logits = model.pair_logits(
                        pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        temperature=Config.TEMPERATURE,
                    )
                    similarity_prob = torch.sigmoid(logits).item() * 100.0

                    # Debug cosine similarity (raw CLIP alignment score)
                    image_embeds = model.encode_images(inputs["pixel_values"])
                    text_embeds = model.encode_text(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    cosine_sim = (image_embeds * text_embeds).sum(dim=-1).item()

                similarity_prob = _calibrate_probability(similarity_prob, cosine_sim)

            except Exception as e:
                error = f"Error during inference: {str(e)}"
            
            return render_template(
                'index.html', 
                prompt=selected_label,
                model_prompt=model_prompt,
                image_path=filename, 
                score=f"{similarity_prob:.2f}%",
                error=error,
                labels=EMOTION_LABELS,
                selected_label=selected_label
            )
            
    return render_template('index.html', labels=EMOTION_LABELS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
