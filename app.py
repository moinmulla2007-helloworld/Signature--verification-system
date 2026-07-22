import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    if 'ref_img' not in request.files or 'test_img' not in request.files:
        return redirect(url_for('index'))
        
    ref_file = request.files['ref_img']
    test_file = request.files['test_img']

    if ref_file.filename == '' or test_file.filename == '':
        return redirect(url_for('index'))
    
    # Clean filenames
    ref_filename = 'ref_' + ref_file.filename
    test_filename = 'test_' + test_file.filename

    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
    
    ref_file.save(ref_path)
    test_file.save(test_path)
    
    img_a = preprocess_image(ref_path)
    img_b = preprocess_image(test_path)
    
    if img_a is None or img_b is None:
        return "Invalid image format uploaded.", 400
        
    # Calculate Euclidean distance
    distance = float(siamese.predict([img_a, img_b])[0][0])
    threshold = 0.20
    is_match = distance < threshold
    similarity = max(0.0, min(100.0, (1.0 - (distance/(threshold * 2))) * 100))
    
    return render_template(
        'result.html',
        match=is_match,
        distance=round(distance, 4),
        similarity=round(similarity, 2),
        ref_filename=ref_filename,
        test_filename=test_filename,
        status='GENUINE MATCH' if is_match else 'FORGERY DETECTED'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)