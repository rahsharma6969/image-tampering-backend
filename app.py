import os
import io
import json
import torch
import logging
import datetime
import numpy as np
from PIL import Image, ExifTags
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model.mantranet import pre_trained_model
from model.utils import convert_pdf_to_images

# === CONFIGURATION ===
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
DEFAULT_PORT = 5000

# === INITIALIZATION ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Enable CORS
CORS(app, origins=[
    "http://localhost:5173",
    "https://image-tampered-frontend.onrender.com"
])

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# === DEVICE & MODEL LOADING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    model = pre_trained_model(weight_path="./model/MantraNetv4.pt")
    model.to(device).eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

# === UTILITY FUNCTIONS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_static_files():
    for f in os.listdir(STATIC_FOLDER):
        if f.startswith('result_') or f.endswith(".png"):
            try:
                os.remove(os.path.join(STATIC_FOLDER, f))
            except Exception:
                pass

def get_enhanced_timestamps(file_storage, is_pdf=False):
    timestamps = {'upload_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if not is_pdf:
        try:
            file_storage.seek(0)
            img = Image.open(file_storage)
            timestamps['image_info'] = {
                'size': f"{img.size[0]}x{img.size[1]}",
                'format': img.format,
                'mode': img.mode
            }
            exif = img._getexif()
            if exif:
                exif_data = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized', 'Make', 'Model', 'Software']:
                        exif_data[tag] = str(value)
                timestamps['EXIF'] = exif_data if exif_data else "EXIF present but no readable fields"
            else:
                timestamps['EXIF'] = "No EXIF metadata found"
        except Exception as e:
            timestamps['metadata_error'] = str(e)
    return timestamps

# === ROUTES ===
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "MantraNet Forgery Detection API",
        "version": "1.0",
        "device": str(device),
        "model_loaded": model is not None
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        cleanup_static_files()
        filename = secure_filename(file.filename.lower())
        logger.info(f"Analyzing file: {filename}")
        is_pdf = filename.endswith(".pdf")

        if is_pdf:
            images = convert_pdf_to_images(file)
        else:
            file.seek(0)
            image = Image.open(file).convert("RGB")
            original_size = image.size
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            images = [image]
            logger.info(f"Image resized from {original_size} to {image.size}")

        file.seek(0)
        timestamps = get_enhanced_timestamps(file, is_pdf=is_pdf)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    scores, results, processing_info = [], [], []

    for i, img in enumerate(images):
        try:
            arr = np.array(img)
            tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)

            output_arr = output[0][0].cpu().numpy()
            mask = (output_arr > 0.2).astype(np.uint8)
            score = 100 * np.sum(mask) / mask.size
            scores.append(score)

            heatmap = Image.fromarray((output_arr * 255).astype(np.uint8)).resize(img.size).convert("RGB")
            red_overlay = Image.new("RGB", img.size, (255, 0, 0))
            overlay_mask = heatmap.convert("L")
            overlay = Image.blend(img, red_overlay, alpha=0.3)
            overlay.putalpha(overlay_mask)
            composite = Image.alpha_composite(img.convert("RGBA"), overlay)

            heatmap_path = os.path.join(STATIC_FOLDER, f"heatmap_{i+1}.png")
            overlay_path = os.path.join(STATIC_FOLDER, f"overlay_{i+1}.png")
            original_path = os.path.join(STATIC_FOLDER, f"original_{i+1}.png")

            img.save(original_path)
            heatmap.save(heatmap_path)
            composite.save(overlay_path)

            results.append({
                "page": i + 1,
                "original": original_path,
                "heatmap": heatmap_path,
                "overlay": overlay_path
            })

            processing_info.append({
                "page": i + 1,
                "shape": list(arr.shape),
                "score": f"{score:.2f}%"
            })
        except Exception as e:
            logger.error(f"Detection failed on page {i+1}: {e}")
            return jsonify({"error": f"Failed on page {i+1}: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "pages": len(images),
        "forgery_scores": [f"{s:.2f}%" for s in scores],
        "timestamps": timestamps,
        "results": results,
        "summary": {
            "max_score": f"{max(scores):.2f}%" if scores else "0.00%",
            "avg_score": f"{sum(scores)/len(scores):.2f}%" if scores else "0.00%",
            "threshold": "20.00%",
            "verdict": "SUSPICIOUS" if max(scores) > 20 else "CLEAN"
        },
        "processing_info": processing_info
    })

@app.route("/results/<path:filename>")
def serve_static(filename):
    try:
        return send_file(os.path.join(STATIC_FOLDER, filename))
    except Exception:
        return jsonify({"error": "File not found"}), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Max 16MB allowed."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# === ENTRYPOINT ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))  # For Render compatibility
    logger.info(f"Starting app on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
