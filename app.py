
from flask import Flask, request, jsonify, send_file
from model.mantranet import pre_trained_model
from model.utils import convert_pdf_to_images
import os
from PIL import Image, ExifTags
import torch
import numpy as np
import datetime
import json
import logging
from werkzeug.utils import secure_filename
import io
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173", 
    "https://image-tampered-frontend.onrender.com"
])


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model
try:
    model = pre_trained_model(weight_path='./model/MantraNetv4.pt')
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_enhanced_timestamps(file_storage, is_pdf=False):
    """Enhanced metadata extraction function"""
    timestamps = {}
    
    # Upload time
    timestamps['upload_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not is_pdf:
        try:
            # Reset file pointer
            file_storage.seek(0)
            img = Image.open(file_storage)
            
            # Basic image info
            timestamps['image_info'] = {
                'size': f"{img.size[0]}x{img.size[1]}",
                'format': img.format,
                'mode': img.mode
            }
            
            # Extract EXIF data
            exif = img._getexif()
            if exif:
                exif_data = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    
                    # DateTime fields
                    if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        exif_data[f"{tag}"] = str(value)
                    # Camera info
                    elif tag in ['Make', 'Model', 'Software']:
                        exif_data[f"{tag}"] = str(value)
                    # Technical details
                    elif tag in ['ExposureTime', 'FNumber', 'ISO', 'FocalLength']:
                        exif_data[f"{tag}"] = str(value)
                    # GPS info
                    elif tag in ['GPSInfo']:
                        exif_data['GPS'] = "GPS data present"
                
                if exif_data:
                    timestamps['EXIF'] = exif_data
                else:
                    timestamps['EXIF'] = "EXIF data present but no relevant fields found"
            else:
                timestamps['EXIF'] = "No EXIF metadata found"
                
        except Exception as e:
            timestamps['metadata_error'] = str(e)
            logger.error(f"Error extracting metadata: {e}")
    
    return timestamps

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)

def cleanup_static_files():
    """Clean up previous result files"""
    try:
        for f in os.listdir(STATIC_FOLDER):
            if f.startswith('result_'):
                os.remove(os.path.join(STATIC_FOLDER, f))
    except Exception as e:
        logger.warning(f"Failed to cleanup static files: {e}")

@app.route('/', methods=['GET'])
def index():
    """Basic info endpoint"""
    return jsonify({
        "service": "MantraNet Forgery Detection API",
        "version": "1.0",
        "device": str(device),
        "model_loaded": model is not None,
        "endpoints": {
            "detect": "/detect (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/detect', methods=['POST'])
def detect():
    """Main forgery detection endpoint with overlay and heatmap."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename.lower())
    logger.info(f"Processing file: {filename}")

    try:
        ensure_directories()
        cleanup_static_files()

        if filename.endswith(".pdf"):
            images_to_check = convert_pdf_to_images(file)
            is_pdf = True
        else:
            file.seek(0)
            image = Image.open(file).convert('RGB')
            original_size = image.size
            MAX_DIM = 768
            image.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
            images_to_check = [image]
            is_pdf = False
            logger.info(f"Image resized from {original_size} to {image.size}")

        file.seek(0)
        timestamps = get_enhanced_timestamps(file, is_pdf=is_pdf)

    except Exception as e:
        logger.error(f"Failed to process image/PDF: {e}")
        return jsonify({"error": f"Failed to read image/PDF: {str(e)}"}), 500

    scores = []
    results = []
    processing_info = []

    for i, image in enumerate(images_to_check):
        try:
            original_array = np.array(image)
            logger.info(f"Processing image {i+1}/{len(images_to_check)}, shape: {original_array.shape}")

            im_tensor = torch.Tensor(original_array).unsqueeze(0).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                final_output = model(im_tensor)

            output_array = final_output[0][0].cpu().detach().numpy()
            mask = (output_array > 0.2).astype(np.uint8)
            forgery_score = 100 * np.sum(mask) / mask.size
            scores.append(forgery_score)

            # === Save heatmap ===
            heatmap_img = Image.fromarray((output_array * 255).astype(np.uint8)).resize(image.size).convert("RGB")
            heatmap_path = f"{STATIC_FOLDER}/heatmap_page_{i+1}.png"
            heatmap_img.save(heatmap_path)

            # === Save original ===
            original_path = f"{STATIC_FOLDER}/original_page_{i+1}.png"
            image.save(original_path)

            # === Create overlay ===
            red_overlay = Image.new("RGB", image.size, (255, 0, 0))
            mask_img = heatmap_img.convert("L")
            overlay = Image.blend(image, red_overlay, alpha=0.3)
            overlay.putalpha(mask_img)
            composite = Image.alpha_composite(image.convert("RGBA"), overlay)
            overlay_path = f"{STATIC_FOLDER}/overlay_page_{i+1}.png"
            composite.save(overlay_path)

            results.append({
                "page": i + 1,
                "original": original_path,
                "heatmap": heatmap_path,
                "overlay": overlay_path
            })

            processing_info.append({
                "page": i + 1,
                "original_shape": original_array.shape,
                "tensor_shape": list(im_tensor.shape),
                "tensor_range": [float(im_tensor.min()), float(im_tensor.max())],
                "output_range": [float(output_array.min()), float(output_array.max())],
                "mask_pixels": int(np.sum(mask)),
                "total_pixels": int(mask.size)
            })

        except Exception as e:
            logger.error(f"Error analyzing image page {i+1}: {e}")
            return jsonify({"error": f"Error analyzing image page {i+1}: {str(e)}"}), 500

    response = {
        "status": "success",
        "pages_analyzed": len(images_to_check),
        "forgery_scores": [f"{s:.2f}%" for s in scores],
        "result_images": results,
        "timestamps": timestamps,
        "processing_info": processing_info,
        "summary": {
            "max_score": f"{max(scores):.2f}%" if scores else "0.00%",
            "avg_score": f"{sum(scores)/len(scores):.2f}%" if scores else "0.00%",
            "suspicious_threshold": "20.00%",
            "verdict": "SUSPICIOUS" if max(scores) > 20 else "CLEAN"
        }
    }

    logger.info(f"Analysis complete. Max score: {max(scores):.2f}%")
    return jsonify(response)

@app.route('/results/<path:filename>')
def serve_result(filename):
    """Serve result images"""
    try:
        return send_file(os.path.join(STATIC_FOLDER, filename))
    except Exception as e:
        return jsonify({"error": "File not found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting MantraNet Forgery Detection API...")
    app.run(debug=True, host='0.0.0.0', port=5000)   