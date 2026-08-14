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
    siamese.load_weights(model_path)
    print(f"Loaded model weights from {model_path}")
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

    # Crop tightly to signature ink
    img = crop_to_ink(img)

    if img.size == 0:
        return np.zeros((1, 256, 256, 1), dtype="float32")

    # Anti-Distortion Aspect Ratio Padding
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

    # Resize to model input dimensions
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Save debug image to static uploads for inspection
    debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"DEBUG_AI_{os.path.basename(path)}")
    cv2.imwrite(debug_path, img)

    # Convert to normalized float32 tensor of shape (1, 256, 256, 1)
    img = img.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)


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

    crop_a = crop_to_ink(raw_a)
    crop_b = crop_to_ink(raw_b)

    if crop_a.size == 0 or crop_b.size == 0:
        return

    def resize_keep_aspect(image, target_h=256):
        h, w = image.shape
        if h == 0:
            return image
        aspect = w / h
        target_w = max(1, int(target_h * aspect))
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    sig_a = resize_keep_aspect(crop_a, target_h=256)
    sig_b = resize_keep_aspect(crop_b, target_h=256)

    max_w = max(sig_a.shape[1], sig_b.shape[1])

    def pad_to_width(image, target_w):
        h, w = image.shape
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            return cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image

    sig_a = pad_to_width(sig_a, max_w)
    sig_b = pad_to_width(sig_b, max_w)

    overlap = cv2.bitwise_and(sig_a, sig_b)
    diff = cv2.absdiff(sig_a, sig_b)

    xai_map = np.zeros((256, max_w, 3), dtype=np.uint8)
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