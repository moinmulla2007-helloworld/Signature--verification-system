import os
import cv2
import time
import uuid
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from model import build_siamese_network, contrastive_loss

app = Flask(__name__)

# Configure upload directory inside static
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Siamese Model
siamese = build_siamese_network()
model_path = 'saved_model/siamese_model.weights.h5'

if os.path.exists(model_path):
    siamese.load_weights(model_path)
    print("Siamese Model loaded successfully.")
else:
    print(f"Warning: {model_path} not found. Run train_siamese.py first.")

def preprocess_image(path):
    """Prepares image for the Siamese Neural Network (128x128 binary)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def crop_to_signature(img_gray):
    """Auxiliary helper to crop white-space padding and align signatures."""
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return thresh
    x, y, w, h = cv2.boundingRect(coords)
    return thresh[y:y+h, x:x+w]

def generate_diff_heatmap(ref_path, test_path, output_path):
    """
    High-Precision Forensic Map:
    - Auto-crops signatures to align stroke bounding boxes
    - Renders at High-Res (512x256)
    - GREEN: Matching stroke structure
    - RED: Structural deviations / Forgeries
    """
    raw_a = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    raw_b = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if raw_a is None or raw_b is None:
        return

    crop_a = crop_to_signature(raw_a)
    crop_b = crop_to_signature(raw_b)

    target_w, target_h = 512, 256
    sig_a = cv2.resize(crop_a, (target_w, target_h), interpolation=cv2.INTER_AREA)
    sig_b = cv2.resize(crop_b, (target_w, target_h), interpolation=cv2.INTER_AREA)

    overlap = cv2.bitwise_and(sig_a, sig_b)
    diff = cv2.absdiff(sig_a, sig_b)

    xai_map = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Color mapping [B, G, R]: Glowing Emerald Green for Match, Crimson Red for Discrepancy
    xai_map[overlap > 40] = [80, 240, 100]  
    xai_map[diff > 40] = [50, 50, 255]     

    xai_map = cv2.GaussianBlur(xai_map, (3, 3), 0)
    cv2.imwrite(output_path, xai_map)

def cleanup_old_files():
    """Removes image files older than 1 hour to manage server storage."""
    current_time = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path)) > 3600:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def save_base64_image(base64_str, output_path):
    """Helper function to decode drawing canvas base64 data to image file."""
    if ',' in base64_str:
        header, data = base64_str.split(',', 1)
    else:
        data = base64_str
    image_bytes = base64.b64decode(data)
    with open(output_path, 'wb') as f:
        f.write(image_bytes)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    cleanup_old_files()

    session_id = uuid.uuid4().hex[:8]
    ref_filename = f"ref_{session_id}.png"
    test_filename = f"test_{session_id}.png"
    heatmap_filename = f"heatmap_{session_id}.png"

    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)

    # Check Reference Mode (Upload File vs Draw Canvas)
    ref_mode = request.form.get('ref_mode', 'file')
    if ref_mode == 'draw':
        ref_draw_data = request.form.get('ref_draw_data', '')
        if not ref_draw_data:
            return redirect(url_for('index'))
        save_base64_image(ref_draw_data, ref_path)
    else:
        if 'ref_img' not in request.files or request.files['ref_img'].filename == '':
            return redirect(url_for('index'))
        ref_file = request.files['ref_img']
        ref_file.save(ref_path)

    # Check Test Mode (Upload File vs Draw Canvas)
    test_mode = request.form.get('test_mode', 'file')
    if test_mode == 'draw':
        test_draw_data = request.form.get('test_draw_data', '')
        if not test_draw_data:
            return redirect(url_for('index'))
        save_base64_image(test_draw_data, test_path)
    else:
        if 'test_img' not in request.files or request.files['test_img'].filename == '':
            return redirect(url_for('index'))
        test_file = request.files['test_img']
        test_file.save(test_path)

    img_a = preprocess_image(ref_path)
    img_b = preprocess_image(test_path)

    if img_a is None or img_b is None:
        return "Invalid image format.", 400

    # Calculate Euclidean distance
    distance = float(siamese.predict([img_a, img_b])[0][0])
    threshold = 0.15
    is_match = distance < threshold
    similarity = max(0.0, min(100.0, (1.0 - (distance / (threshold * 2))) * 100))

    # Generate Explainable AI High-Res Heatmap
    generate_diff_heatmap(ref_path, test_path, heatmap_path)

    return render_template(
        'result.html',
        match=is_match,
        distance=round(distance, 4),
        similarity=round(similarity, 1),
        ref_filename=ref_filename,
        test_filename=test_filename,
        heatmap_filename=heatmap_filename,
        status='GENUINE MATCH' if is_match else 'FORGERY DETECTED'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)