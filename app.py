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
from preprocessing import (
    load_and_standardize, crop_and_center, clean_ink_mask,
    preprocess_for_model
)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD SIAMESE NETWORK
# ---------------------------------------------------------
siamese = build_siamese_network()
model_path = os.path.join('saved_model', 'siamese_model.weights.h5')

if os.path.exists(model_path):
    try:
        siamese.load_weights(model_path)
        print(f"Loaded model weights from {model_path}")
    except ValueError as e:
        # This fires if model_path was saved from a different architecture
        # (e.g. the pre-BatchNorm version of model.py). Predictions will be
        # untrained/random until you retrain and re-save weights against
        # the CURRENT model.py - but the app will still boot, which is
        # useful for testing the EXIF/preprocessing pipeline in isolation.
        print(f"Warning: could not load {model_path} - architecture mismatch "
              f"with current model.py. Running with untrained weights.\n{e}")
else:
    print(f"Warning: {model_path} not found. Ensure model is trained first.")

# ---------------------------------------------------------
# 2. IMAGE PREPROCESSING (shared with train.py via preprocessing.py)
# ---------------------------------------------------------
def preprocess_image(path):
    """
    Prepares an uploaded image for Siamese Network inference. Delegates
    the actual image processing to preprocessing.preprocess_for_model()
    (shared with train.py) and adds the two things that are specific to
    inference: writing a debug snapshot, and shaping the tensor with a
    batch dimension for model.predict().
    """
    img = preprocess_for_model(path, canvas_size=256)
    if img is None:
        return None

    # Save debug image to static uploads for inspection
    debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"DEBUG_AI_{os.path.basename(path)}")
    cv2.imwrite(debug_path, img)

    # Convert to normalized float32 tensor of shape (1, 256, 256, 1)
    img = img.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)


def align_via_phase_correlation(base, moving, max_shift=60):
    """
    Finds the (dx, dy) translation that best overlaps `moving` onto `base`
    using phase correlation on the two ink masks, then returns a shifted
    copy of `moving`.

    crop_and_center() gets both signatures into roughly the same place via
    centroid, but a heavy/asymmetric flourish (a long underline on one
    sample but not the other) pulls the centroid away from the letters by
    a different amount on each side - so the coarse centering alone isn't
    exact. Phase correlation directly computes the shift that maximizes
    cross-correlation between the two ink patterns, which is exactly what
    "overlap the signatures" means mathematically. This is a refinement
    pass on top of crop_and_center, not a replacement for it.

    max_shift caps how far a correction is trusted (in pixels) - phase
    correlation can return a large, spurious shift when two images share
    very little ink in common (e.g. a near-empty crop), so a wild result
    is clamped rather than applied outright.
    """
    base_f = base.astype(np.float32)
    moving_f = moving.astype(np.float32)

    if base_f.sum() == 0 or moving_f.sum() == 0:
        return moving

    try:
        (dx, dy), _response = cv2.phaseCorrelate(base_f, moving_f)
    except cv2.error:
        return moving

    dx = float(np.clip(dx, -max_shift, max_shift))
    dy = float(np.clip(dy, -max_shift, max_shift))

    shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(
        moving, shift_matrix, (moving.shape[1], moving.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=0
    )
    return aligned


# ---------------------------------------------------------
# 4. DIFFERENCE HEATMAP GENERATION
# ---------------------------------------------------------
def generate_diff_heatmap(ref_path, test_path, output_path):
    """
    Generates a visual comparison map:
    - Green = Overlapping / matching strokes
    - Red   = Divergent / non-overlapping strokes
    """
    raw_a = load_and_standardize(ref_path)
    raw_b = load_and_standardize(test_path)

    if raw_a is None or raw_b is None:
        return

    raw_a = cv2.bitwise_not(raw_a)
    raw_b = cv2.bitwise_not(raw_b)

    # Same noise cleanup used by preprocess_image, so the heatmap shows
    # exactly what the model is scoring - not a torn-edge-corrupted crop.
    raw_a = clean_ink_mask(raw_a)
    raw_b = clean_ink_mask(raw_b)

    # Same centroid-centered crop/scale used for the model input, so the
    # heatmap shows what the network is comparing - and so the two
    # signatures' letters land in roughly the same place before refinement.
    canvas_size = 256
    sig_a = crop_and_center(raw_a, canvas_size=canvas_size, ink_fill_ratio=0.8)
    sig_b = crop_and_center(raw_b, canvas_size=canvas_size, ink_fill_ratio=0.8)

    # Refine the coarse centroid alignment into the tightest possible pixel
    # overlap. sig_a (reference) is treated as the fixed anchor; sig_b
    # (test) is the one that gets shifted onto it.
    sig_b = align_via_phase_correlation(sig_a, sig_b)

    overlap = cv2.bitwise_and(sig_a, sig_b)
    diff = cv2.absdiff(sig_a, sig_b)

    xai_map = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    xai_map[overlap > 0] = [80, 240, 100]  # Green for matching strokes
    xai_map[diff > 0] = [50, 50, 255]      # Red for divergent strokes

    xai_map = cv2.GaussianBlur(xai_map, (3, 3), 0)
    cv2.imwrite(output_path, xai_map)


# ---------------------------------------------------------
# 5. UTILITIES & ROUTES
# ---------------------------------------------------------
def cleanup_old_files():
    """Removes uploads older than 1 hour to prevent disk bloat."""
    current_time = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path)) > 3600:
            try:
                os.remove(file_path)
            except Exception:
                pass


def save_base64_image(base64_str, output_path):
    """Decodes canvas base64 signature strings and writes to disk."""
    if ',' in base64_str:
        _, data = base64_str.split(',', 1)
    else:
        data = base64_str
    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(data))


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

    # 1. Process Reference Image Input
    if request.form.get('ref_mode', 'file') == 'draw':
        data = request.form.get('ref_draw_data', '')
        if not data:
            return redirect(url_for('index'))
        save_base64_image(data, ref_path)
    else:
        if 'ref_img' not in request.files or request.files['ref_img'].filename == '':
            return redirect(url_for('index'))
        request.files['ref_img'].save(ref_path)

    # 2. Process Test Image Input
    if request.form.get('test_mode', 'file') == 'draw':
        data = request.form.get('test_draw_data', '')
        if not data:
            return redirect(url_for('index'))
        save_base64_image(data, test_path)
    else:
        if 'test_img' not in request.files or request.files['test_img'].filename == '':
            return redirect(url_for('index'))
        request.files['test_img'].save(test_path)

    # 3. Preprocess both images
    img_a = preprocess_image(ref_path)
    img_b = preprocess_image(test_path)

    if img_a is None or img_b is None:
        return "Invalid image format.", 400

    # 4. Predict Distance
    distance = float(siamese.predict([img_a, img_b])[0][0])

    # 5. Threshold Comparison
    # NOTE: 0.30 is stricter than the one confirmed genuine-pair reading
    # observed during testing (0.3578) - that pair would still be flagged
    # as a forgery at this threshold. If genuine signatures keep getting
    # rejected, this is the first place to look.
    threshold = 0.30
    is_match = distance < threshold
    similarity = max(0.0, min(100.0, (1.0 - (distance / (threshold * 1.5))) * 100))

    # 6. Generate Difference Heatmap
    generate_diff_heatmap(ref_path, test_path, heatmap_path)

    return render_template(
        'result.html',
        match=is_match,
        distance=round(distance, 4),
        similarity=round(similarity, 1),
        threshold=threshold,
        ref_filename=ref_filename,
        test_filename=test_filename,
        heatmap_filename=heatmap_filename,
        status='GENUINE MATCH' if is_match else 'FORGERY DETECTED'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)