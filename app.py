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

def load_and_standardize(path):
    """
    Forces every image into pure binary (Black Ink on White Background).
    Uses Adaptive Thresholding to destroy camera shadows and gray paper.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None

    # 1. Handle Web Canvas (Transparent Background)
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        binary = np.ones(alpha.shape, dtype=np.uint8) * 255 
        binary[alpha > 0] = 0 
        return binary

    # 2. Convert to Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Handle Camera Photos (Adaptive Thresholding)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 15
    )
    
    return binary

def preprocess_image(path):
    """Prepares image for Siamese Network."""
    img = load_and_standardize(path)
    if img is None:
        return None

    # Invert to White Ink on Black Background for the neural network
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Save debug copy
    debug_filename = f"DEBUG_AI_{os.path.basename(path)}"
    debug_path = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
    cv2.imwrite(debug_path, img)

    # Normalize for the neural network
    img = img.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

def crop_to_signature(img_binary_inv):
    """Simplified cropper: Assumes image is already White Ink on Black Bg."""
    coords = cv2.findNonZero(img_binary_inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img_binary_inv[y:y+h, x:x+w]
    return img_binary_inv

def generate_diff_heatmap(ref_path, test_path, output_path):
    """Generates XAI Map utilizing the foolproof standardizer."""
    raw_a = load_and_standardize(ref_path)
    raw_b = load_and_standardize(test_path)

    if raw_a is None or raw_b is None:
        return

    # Invert to White Ink on Black Background for heatmap logic
    raw_a = cv2.bitwise_not(raw_a)
    raw_b = cv2.bitwise_not(raw_b)

    crop_a = crop_to_signature(raw_a)
    crop_b = crop_to_signature(raw_b)

    target_w, target_h = 512, 256
    
    if crop_a.size == 0 or crop_b.size == 0:
        return
        
    sig_a = cv2.resize(crop_a, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    sig_b = cv2.resize(crop_b, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    overlap = cv2.bitwise_and(sig_a, sig_b)
    diff = cv2.absdiff(sig_a, sig_b)

    xai_map = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    xai_map[overlap > 0] = [80, 240, 100]  
    xai_map[diff > 0] = [50, 50, 255]     

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

    # Check Reference Mode
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

    # Check Test Mode
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