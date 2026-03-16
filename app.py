"""
app.py — Flask REST API for Cloud Removal Inference
POST /predict  →  returns URL of denoised image

Usage:
    python app.py
    curl -X POST http://localhost:5000/predict -F "file=@cloudy.jpg"
"""

import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms

# ── Import model from src/ ─────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model import Generator

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

STATIC_DIR    = os.path.join(os.path.dirname(__file__), 'static')
MODEL_PATH    = os.environ.get('MODEL_PATH', 'outputs/generator.pth')
IMG_SIZE      = int(os.environ.get('IMG_SIZE', 64))
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(STATIC_DIR, exist_ok=True)

# ── Load Model Once at Startup ────────────────────────────────────────────────
generator = Generator().to(DEVICE)
try:
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()
    print(f"[API] Generator loaded from '{MODEL_PATH}' on {DEVICE}")
except FileNotFoundError:
    print(f"[API] WARNING: Model file not found at '{MODEL_PATH}'. "
          "Train first with: python src/train.py")

# ── Transforms ────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def tensor_to_pil(tensor):
    """Convert a (C, H, W) float tensor to a PIL Image."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a .jpg/.png file and returns a denoised satellite image.

    Form data:
        file  (required): image file to denoise

    Response JSON:
        output_img_url: path to download the generated image
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file field in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = generator(img_tensor).squeeze(0)

    output_pil  = tensor_to_pil(output)
    output_name = 'generated_image.png'
    output_path = os.path.join(STATIC_DIR, output_name)
    output_pil.save(output_path)

    return jsonify({'output_img_url': f'/static/{output_name}'})


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
