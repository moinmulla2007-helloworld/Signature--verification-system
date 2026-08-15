import os
import cv2
import time
import uuid
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from model import build_siamese_network, contrastive_loss

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
# 2. EXIF-SAFE IMAGE READING
# ---------------------------------------------------------
def read_image_exif_safe(path):
    """
    Reads an image the way a human (and a browser <img> tag) sees it.

    cv2.imread() ignores the EXIF Orientation tag entirely, so a portrait
    phone photo stored with Orientation=6/8/3 comes in still rotated on
    its side even though the frontend displays it upright. This function
    opens the file with Pillow, physically bakes the EXIF rotation into
    the pixel data with ImageOps.exif_transpose(), and only then converts
    to a numpy array in the channel order OpenCV expects.

    Returns a numpy array: grayscale (H,W), BGR (H,W,3), or BGRA (H,W,4) -
    same shape conventions load_and_standardize() already handles.
    """
    pil_img = Image.open(path)

    # This is the actual fix: rotates/flips pixel data according to the
    # EXIF Orientation tag, then strips the tag so nothing downstream
    # (cv2, browsers, debug image writes) can double-apply it.
    pil_img = ImageOps.exif_transpose(pil_img)

    mode = pil_img.mode
    if mode == 'RGBA':
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif mode == 'RGB':
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif mode in ('L', 'LA', '1'):
        gray = pil_img.convert('L')
        return np.array(gray)
    else:
        # P (palette), CMYK, etc. - normalize to RGB first
        arr = np.array(pil_img.convert('RGB'))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------
# 3. IMAGE PREPROCESSING PIPELINE
# ---------------------------------------------------------
def load_and_standardize(path):
    """
    Reads input image (canvas drawing, scanned doc, or phone camera photo),
    applies EXIF-safe orientation correction, removes shadows/lighting
    gradients, and binarizes to pure black ink on white background.
    """
    img = read_image_exif_safe(path)
    if img is None:
        return None

    # Handle transparent digital canvas signatures (RGBA)
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        binary = np.ones(alpha.shape, dtype=np.uint8) * 255
        binary[alpha > 0] = 0
        return binary

    # Convert BGR to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Invert if dark background with light ink
    if np.median(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Illumination Normalization (Shadow Removal via Gaussian division)
    bg_map = cv2.GaussianBlur(gray, (75, 75), 0)
    bg_map = np.clip(bg_map, 1, 255)
    normalized = cv2.divide(gray, bg_map, scale=255)

    # Otsu Binarization (Strict White Background, Black Ink)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def crop_to_ink(img):
    """
    Tightly crops around all valid ink strokes, ignoring tiny isolated specks.
    Expects ink = 255 (white), background = 0 (black).

    Deliberately axis-aligned only (cv2.boundingRect on findNonZero).
    NOTE: do NOT replace this with cv2.minAreaRect-based deskewing - its
    returned angle has an inherent 90/180-degree ambiguity for near-square
    or symmetric ink blobs, which is what was flipping signatures before.
    Now that EXIF rotation is corrected at read time, remaining skew is
    just natural handwriting slant, which the network should learn to
    handle from the CEDAR training distribution rather than have it
    "corrected" away geometrically.
    """
    coords = cv2.findNonZero(img)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    if w > 10 and h > 10:
        return img[y:y+h, x:x+w]
    return img


def crop_and_center(img, canvas_size=256, ink_fill_ratio=0.8):
    """
    Tightly crops to ink, uniformly scales (aspect-preserving) so the ink's
    larger dimension fills `ink_fill_ratio` of the canvas, then places it on
    a canvas_size x canvas_size canvas centered on the ink's CENTROID
    (center of mass) rather than the bounding-box center.

    Why centroid and not bounding-box center: a bounding box gets pulled
    off-center by any asymmetric stroke - e.g. a long underline flourish on
    one signature but not the other. Two crops of the same name can end up
    with their *letters* sitting in noticeably different spots even though
    both boxes are individually "centered." Centering on center-of-mass
    keeps the bulk of the ink (the actual letterforms) anchored to the same
    canvas position across different samples, which is what makes two
    signatures of the same name overlap closely in the diff heatmap and
    gives the Siamese network a consistent, comparable input.

    Used identically by preprocess_image() (model input) and
    generate_diff_heatmap() (visualization) so what the model sees and what
    the heatmap shows are the same alignment.

    Expects ink = 255 (white), background = 0 (black).
    """
    cropped = crop_to_ink(img)
    if cropped.size == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    h, w = cropped.shape
    scale = (canvas_size * ink_fill_ratio) / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center of mass of the ink pixels in the resized crop
    moments = cv2.moments(resized, binaryImage=True)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = new_w / 2.0, new_h / 2.0

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Shift so (cx, cy) lands exactly at the canvas center
    offset_x = int(round(canvas_size / 2.0 - cx))
    offset_y = int(round(canvas_size / 2.0 - cy))

    # Clip to the overlapping region so a large offset (very asymmetric
    # ink) can never index outside the canvas or the source crop.
    src_x0, src_y0 = max(0, -offset_x), max(0, -offset_y)
    dst_x0, dst_y0 = max(0, offset_x), max(0, offset_y)
    copy_w = min(new_w - src_x0, canvas_size - dst_x0)
    copy_h = min(new_h - src_y0, canvas_size - dst_y0)

    if copy_w > 0 and copy_h > 0:
        canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
            resized[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]

    return canvas


def preprocess_image(path):
    """
    Full pipeline to prepare uploaded image for Siamese Network inference:
    1. EXIF-safe read + binarize & remove background shadows
    2. Invert (ink=255, bg=0)
    3. Wipe outer edge shadows
    4. Remove noise contours
    5. Crop tightly around ink (axis-aligned, orientation preserved)
    6. Pad to square ratio & resize to 256x256
    7. Normalize to [0.0, 1.0] float32 tensor
    """
    img = load_and_standardize(path)
    if img is None:
        return None

    # Invert for geometric operations: Ink = 255 (white), Background = 0 (black)
    img = cv2.bitwise_not(img)

    # Outer border shadow wipe
    border = 10
    img[:border, :] = 0
    img[-border:, :] = 0
    img[:, :border] = 0
    img[:, -border:] = 0

    # Noise filter: remove isolated tiny specks
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 25 or w < 3 or h < 3:
            cv2.drawContours(img, [cnt], -1, 0, -1)

    # Crop to ink, scale, and center on the ink's centroid so the letters
    # themselves line up consistently across different signature samples
    # (see crop_and_center docstring for why this replaces plain bbox
    # crop + pad + resize).
    img = crop_and_center(img, canvas_size=256, ink_fill_ratio=0.8)

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

    # 5. Strict 0.20 Threshold Comparison
    threshold = 0.20
    is_match = distance < threshold
    similarity = max(0.0, min(100.0, (1.0 - (distance / (threshold * 1.5))) * 100))

    # 6. Generate Difference Heatmap
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