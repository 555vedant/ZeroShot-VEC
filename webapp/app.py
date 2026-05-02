import os
import sys
import torch
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_werkzeug_filename # Using standard secure_filename

# Add project root to sys path so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import CLIPFineTuner
from transformers import CLIPProcessor
from utils.config import Config
from src.inference import SearchEngine
from src.dataset import resolve_image_path

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

print("Initializing Search Engine (this will pre-compute dataset embeddings)...", flush=True)
search_engine = SearchEngine()
model = search_engine.model
processor = search_engine.processor
device = search_engine.device
print("Search Engine loaded successfully!", flush=True)

@app.route('/dataset_images/<path:filepath>')
def dataset_images(filepath):
    abs_path = resolve_image_path(filepath)
    # Using flask send_file
    from flask import send_file
    return send_file(str(abs_path))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_prompt = request.form.get('prompt', '')
        
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
            top_matches = []
            try:
                # Process inputs
                inputs = processor(text=[query_prompt], images=image, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    # Generate embeddings
                    image_features = model.encode_images(inputs["pixel_values"])
                    text_features = model.encode_text(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"]
                    )
                    
                    # Normalize and compute similarity
                    image_embeds = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # Scale by logit scale if possible, otherwise just raw dot product
                    logit_scale = model.model.logit_scale.exp()
                    sim = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
                    similarity_score = sim.item()
                
                # Fetch top 5 from dataset
                top_matches = search_engine.search(query_prompt, top_k=5)
            except Exception as e:
                error = f"Error during inference: {str(e)}"
            
            return render_template(
                'index.html', 
                prompt=query_prompt, 
                image_path=filename, 
                score=f"{similarity_score:.4f}",
                top_matches=top_matches,
                error=error
            )
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
