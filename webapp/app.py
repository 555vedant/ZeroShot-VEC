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
from src.dataset import format_emotion_prompt, normalize_emotion_text, resolve_image_path
from src.inference import SearchEngine
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

# Lazy search engine (dataset-backed image retrieval)
_search_engine = None
_search_engine_error = None


def _get_search_engine():
    global _search_engine, _search_engine_error
    if _search_engine is not None or _search_engine_error is not None:
        return _search_engine

    try:
        cp_path = None
        if local_checkpoint.exists():
            cp_path = local_checkpoint
        elif config_checkpoint.exists():
            cp_path = config_checkpoint

        dp_path = Path(Config.DATA_FILE)
        if not dp_path.exists():
            raise FileNotFoundError(f"Missing pairs.json at {dp_path}")

        _search_engine = SearchEngine(checkpoint_path=cp_path, data_path=dp_path)
    except Exception as exc:
        _search_engine_error = f"Search disabled: {exc}"

    return _search_engine

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
        raw_prompt = request.form.get('prompt', '').strip()
        
        if not raw_prompt:
            return render_template('index.html', error='No prompt provided.')

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
            similarity_score = 0.0
            model_prompt = raw_prompt
            top_matches = []
            search_error = None
            try:
                normalized = normalize_emotion_text(raw_prompt)
                if normalized:
                    model_prompt = format_emotion_prompt(normalized)

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

                # Top-K image search using dataset embeddings
                engine = _get_search_engine()
                if engine is None:
                    search_error = _search_engine_error
                else:
                    top_matches = engine.search(model_prompt, top_k=Config.SEARCH_TOP_K)
                
            except Exception as e:
                error = f"Error during inference: {str(e)}"
            
            return render_template(
                'index.html', 
                prompt=raw_prompt,
                model_prompt=model_prompt,
                image_path=filename, 
                score=f"{similarity_prob:.2f}%",
                raw_cosine=f"{cosine_sim:.4f}",
                error=error,
                top_matches=top_matches,
                search_error=search_error
            )
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
