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
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

siamese = build_siamese_network()
model_path = 'saved_model/siamese_model.weights.h5'

if os.path.exists(model_path):
    siamese.load_weights(model_path)
    print("Siamese Model loaded successfully.")
else:
    print(f"Warning: {model_path} not found. Run train_siamese.py first.")

def load_and_standardize(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        binary = np.ones(alpha.shape, dtype=np.uint8) * 255 
        binary[alpha > 0] = 0 
        return binary

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if np.median(blurred) < 127:
        blurred = cv2.bitwise_not(blurred)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 15
    )
    return binary

def preprocess_image(path):
    img = load_and_standardize(path)
    if img is None: return None

    img = cv2.bitwise_not(img)

    # 10-pixel border wipe for camera shadows
    border = 10
    img[:border, :] = 0
    img[-border:, :] = 0
    img[:, :border] = 0
    img[:, -border:] = 0

    # Contour Noise Filtering
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 25 or w < 3 or h < 3:
            cv2.drawContours(img, [cnt], -1, 0, -1)

    # Tight Crop to clean ink
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 10 and h > 10:  
            img = img[y:y+h, x:x+w]

    if img.size == 0:
        return np.zeros((1, 128, 128, 1), dtype="float32")

    # Anti-Distortion Padding
    h, w = img.shape
    diff = abs(h - w)
    if h > w:
        pad_left = diff // 2
        pad_right = diff - pad_left
        img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Safely resize
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Save debug copy for website display
    debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"DEBUG_AI_{os.path.basename(path)}")
    cv2.imwrite(debug_path, img)

    # Normalize for AI
    img = img.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

def crop_to_signature(img_binary_inv):
    coords = cv2.findNonZero(img_binary_inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img_binary_inv[y:y+h, x:x+w]
    return img_binary_inv

def generate_diff_heatmap(ref_path, test_path, output_path):
    raw_a = load_and_standardize(ref_path)
    raw_b = load_and_standardize(test_path)

    if raw_a is None or raw_b is None: return

    raw_a = cv2.bitwise_not(raw_a)
    raw_b = cv2.bitwise_not(raw_b)

    crop_a = crop_to_signature(raw_a)
    crop_b = crop_to_signature(raw_b)
    
    if crop_a.size == 0 or crop_b.size == 0: return
        
    sig_a = cv2.resize(crop_a, (512, 256), interpolation=cv2.INTER_NEAREST)
    sig_b = cv2.resize(crop_b, (512, 256), interpolation=cv2.INTER_NEAREST)

    overlap = cv2.bitwise_and(sig_a, sig_b)
    diff = cv2.absdiff(sig_a, sig_b)

    xai_map = np.zeros((256, 512, 3), dtype=np.uint8)
    xai_map[overlap > 0] = [80, 240, 100]  
    xai_map[diff > 0] = [50, 50, 255]     

    xai_map = cv2.GaussianBlur(xai_map, (3, 3), 0)
    cv2.imwrite(output_path, xai_map)

def cleanup_old_files():
    current_time = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path)) > 3600:
            try: os.remove(file_path)
            except: pass

def save_base64_image(base64_str, output_path):
    if ',' in base64_str: _, data = base64_str.split(',', 1)
    else: data = base64_str
    with open(output_path, 'wb') as f: f.write(base64.b64decode(data))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    cleanup_old_files()

    session_id = uuid.uuid4().hex[:8]
    ref_filename, test_filename, heatmap_filename = f"ref_{session_id}.png", f"test_{session_id}.png", f"heatmap_{session_id}.png"
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)

    # Reference Image
    if request.form.get('ref_mode', 'file') == 'draw':
        data = request.form.get('ref_draw_data', '')
        if not data: return redirect(url_for('index'))
        save_base64_image(data, ref_path)
    else:
        if 'ref_img' not in request.files or request.files['ref_img'].filename == '': return redirect(url_for('index'))
        request.files['ref_img'].save(ref_path)

    # Test Image
    if request.form.get('test_mode', 'file') == 'draw':
        data = request.form.get('test_draw_data', '')
        if not data: return redirect(url_for('index'))
        save_base64_image(data, test_path)
    else:
        if 'test_img' not in request.files or request.files['test_img'].filename == '': return redirect(url_for('index'))
        request.files['test_img'].save(test_path)

    img_a, img_b = preprocess_image(ref_path), preprocess_image(test_path)
    if img_a is None or img_b is None: return "Invalid image format.", 400

    distance = float(siamese.predict([img_a, img_b])[0][0])
    
    # 0.20 THRESHOLD
    threshold = 0.20
    is_match = distance < threshold
    similarity = max(0.0, min(100.0, (1.0 - (distance / (threshold * 1.5))) * 100))

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