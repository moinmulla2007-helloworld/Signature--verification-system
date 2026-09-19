"""
SignaVerify - Flask app.

Changes from the original version:
  * Uploads are validated with Pillow (type, size, blank check, EXIF rotation)
    and processed in a temporary directory that is deleted immediately.
    Nothing is written to static/ and results are returned as data URIs.
  * Three-state verdict (likely genuine / inconclusive / likely different)
    instead of a hard "FORGERY DETECTED".
  * Threshold is loaded from saved_model/threshold.json (or the
    SIGNAVERIFY_THRESHOLD env var) instead of being hardcoded in the route.
  * Similarity % is scaled so the threshold sits at 50%.
  * The app refuses to give verdicts if the model weights failed to load
    (unless SIGNAVERIFY_ALLOW_UNTRAINED=1 is set for pipeline testing).
  * Phase-correlation shift direction fixed (the old code moved the test
    signature AWAY from the reference).
  * Difference map tolerates small stroke offsets and shows which signature
    the extra ink belongs to.
  * debug mode is opt-in via FLASK_DEBUG=1.
"""
import base64
import binascii
import io
import json
import os
import tempfile

import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from model import build_siamese_network
from preprocessing import (
    load_and_standardize, crop_and_center, clean_ink_mask,
    preprocess_for_model,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'siamese_model.weights.h5')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'saved_model', 'threshold.json')

DEFAULT_THRESHOLD = 0.30      # used only if no threshold.json / env var
GENUINE_FACTOR = 0.85         # distance < threshold * 0.85  -> likely genuine
DIFFERENT_FACTOR = 1.15       # distance > threshold * 1.15  -> likely different

CANVAS_SIZE = 256
MAX_UPLOAD_BYTES = 5 * 1024 * 1024          # per image
MAX_CANVAS_CHARS = 8 * 1024 * 1024          # base64 text of a drawn signature
ALLOW_UNTRAINED = os.environ.get('SIGNAVERIFY_ALLOW_UNTRAINED') == '1'

# Warn at 25 MP, refuse at 50 MP (decompression-bomb protection).
Image.MAX_IMAGE_PIXELS = 25_000_000

# Difference-map colours (BGR for OpenCV). Keep in sync with the legend
# swatches in templates/result.html (ink grey / seal red / blue).
PAPER_BGR = (244, 247, 247)   # #f7f7f4
MATCH_BGR = (70, 63, 59)      # #3b3f46  strokes present in both
REF_BGR = (48, 37, 122)       # #7a2530  ink only in the reference
TEST_BGR = (196, 81, 36)      # #2451c4  ink only in the test sample

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Flask/Werkzeug >= 3.1 caps non-file form data at 500 KB by default, which
# is too small for a base64 canvas drawing. Older versions ignore this key.
app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024


def load_threshold():
    env = os.environ.get('SIGNAVERIFY_THRESHOLD')
    candidates = []
    if env:
        candidates.append(env)
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
                candidates.append(json.load(f).get('threshold'))
        except (OSError, ValueError, AttributeError) as e:
            print(f"Warning: could not read {THRESHOLD_PATH}: {e}")
    for value in candidates:
        try:
            value = float(value)
            if value > 0:
                return value
        except (TypeError, ValueError):
            continue
    return DEFAULT_THRESHOLD


THRESHOLD = load_threshold()

# ---------------------------------------------------------
# 1. LOAD SIAMESE NETWORK
# ---------------------------------------------------------
siamese = build_siamese_network()
MODEL_READY = False

if os.path.exists(MODEL_PATH):
    try:
        siamese.load_weights(MODEL_PATH)
        MODEL_READY = True
        print(f"Loaded model weights from {MODEL_PATH}")
    except (ValueError, OSError) as e:
        # Fires if the weights were saved from a different architecture
        # (e.g. the pre-BatchNorm version of model.py).
        print(f"Warning: could not load {MODEL_PATH} - architecture mismatch "
              f"with current model.py.\n{e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Train the model first.")

if not MODEL_READY:
    if ALLOW_UNTRAINED:
        print("SIGNAVERIFY_ALLOW_UNTRAINED=1: running with untrained weights. "
              "Scores are meaningless.")
    else:
        print("The app will return an error on /verify until weights load.")

print(f"Decision threshold: {THRESHOLD}")


# ---------------------------------------------------------
# 2. INPUT HANDLING
# ---------------------------------------------------------
class InputError(Exception):
    """A problem with what the user submitted (shown to them as-is)."""


def decode_image(raw, label):
    """
    Validates raw bytes as an image and returns a clean RGB PIL image:
    EXIF rotation applied, transparency flattened onto white, and blank
    images rejected. Phone photos can be HEIC, which Pillow can't read
    unless pillow-heif is installed - those get the "couldn't be read" error.
    """
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError,
            OSError, ValueError, SyntaxError):
        raise InputError(
            f"The {label} image couldn't be read. Use a PNG or JPG file.")

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        flat = Image.new('RGB', rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        img = flat
    else:
        img = img.convert('RGB')

    gray = np.asarray(img.convert('L'))
    if int(gray.max()) - int(gray.min()) < 40:
        raise InputError(
            f"The {label} image looks blank. Add a signature with clear contrast.")
    return img


def read_signature(field, label):
    """Reads the reference ('ref') or test ('test') signature from the form."""
    mode = request.form.get(f'{field}_mode', 'file')

    if mode == 'draw':
        data = request.form.get(f'{field}_draw_data', '')
        if not data:
            raise InputError(f"Draw the {label} signature first.")
        if len(data) > MAX_CANVAS_CHARS:
            raise InputError(f"The {label} drawing is too large.")
        if ',' in data:
            data = data.split(',', 1)[1]
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise InputError(f"The {label} drawing couldn't be read. Try drawing it again.")
    else:
        upload = request.files.get(f'{field}_img')
        if upload is None or upload.filename == '':
            raise InputError(f"Add an image for the {label} signature.")
        raw = upload.stream.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise InputError(
                f"The {label} image is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    return decode_image(raw, label)


def image_to_data_uri(img, max_side=900):
    """Downscaled JPEG data URI used to show the submitted images back."""
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# ---------------------------------------------------------
# 3. MODEL INPUT
# ---------------------------------------------------------
def preprocess_image(path):
    """
    Prepares an image for the Siamese network by delegating to
    preprocessing.preprocess_for_model() (shared with train_siamese.py).
    Returns a float32 tensor of shape (1, 256, 256, 1), or None.
    """
    img = preprocess_for_model(path, canvas_size=CANVAS_SIZE)
    if img is None:
        return None
    img = img.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)


def predict_distance(img_a, img_b):
    # Direct call is much faster than model.predict() for a single pair.
    out = siamese([img_a, img_b], training=False)
    return float(np.asarray(out).reshape(-1)[0])


def classify(distance, threshold):
    if distance < threshold * GENUINE_FACTOR:
        return 'genuine'
    if distance > threshold * DIFFERENT_FACTOR:
        return 'different'
    return 'inconclusive'


VERDICT_TEXT = {
    'genuine': (
        'Likely genuine',
        'The test signature is close to the reference.'),
    'inconclusive': (
        'Inconclusive',
        'The score is close to the decision threshold. '
        'Try a clearer image or a better reference.'),
    'different': (
        'Likely different',
        'The test signature differs noticeably from the reference.'),
}


# ---------------------------------------------------------
# 4. DIFFERENCE MAP
# ---------------------------------------------------------
def build_signature_mask(path):
    """Cleaned, centred ink mask (same steps as the model's preprocessing)."""
    raw = load_and_standardize(path)
    if raw is None:
        return None
    raw = cv2.bitwise_not(raw)
    raw = clean_ink_mask(raw)
    return crop_and_center(raw, canvas_size=CANVAS_SIZE, ink_fill_ratio=0.8)


def align_via_phase_correlation(base, moving, max_shift=60):
    """
    Refines the coarse centroid alignment from crop_and_center() by finding
    the translation that best overlaps `moving` onto `base`.

    cv2.phaseCorrelate(base, moving) returns how far `moving` is displaced
    FROM `base`, so the correction is the NEGATIVE of that shift. (The
    previous version applied it with the wrong sign, which doubled the
    misalignment instead of removing it.)

    max_shift clamps the correction: with very little shared ink, phase
    correlation can return a large spurious shift.
    """
    base_f = base.astype(np.float32)
    moving_f = moving.astype(np.float32)

    if base_f.sum() == 0 or moving_f.sum() == 0:
        return moving

    try:
        (dx, dy), _response = cv2.phaseCorrelate(base_f, moving_f)
    except cv2.error:
        return moving

    dx = -float(np.clip(dx, -max_shift, max_shift))
    dy = -float(np.clip(dy, -max_shift, max_shift))

    shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        moving, shift_matrix, (moving.shape[1], moving.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=0
    )


def render_difference_map(sig_a, sig_b, tolerance=5):
    """
    Returns PNG bytes.
      dark grey = ink present in both (within `tolerance` px)
      red       = ink only in the reference
      blue      = ink only in the test sample
    This is a pixel-level comparison after alignment, not an explanation of
    what the neural network attended to.
    """
    sig_b = align_via_phase_correlation(sig_a, sig_b)

    a = sig_a > 0
    b = sig_b > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance))
    near_a = cv2.dilate(a.astype(np.uint8), kernel) > 0
    near_b = cv2.dilate(b.astype(np.uint8), kernel) > 0

    matched = (a & near_b) | (b & near_a)
    ref_only = a & ~near_b
    test_only = b & ~near_a

    h, w = sig_a.shape[:2]
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = PAPER_BGR
    canvas[matched] = MATCH_BGR
    canvas[ref_only] = REF_BGR
    canvas[test_only] = TEST_BGR

    # Signatures are wide, so crop the square canvas to the ink (plus a
    # margin) - otherwise the map is a thin strip in a lot of empty paper.
    ys, xs = np.nonzero(a | b)
    if len(xs):
        pad = 10
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
        canvas = canvas[y0:y1, x0:x1]

    canvas = cv2.resize(canvas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    ok, buf = cv2.imencode('.png', canvas)
    if not ok:
        raise RuntimeError('Could not encode difference map')
    return buf.tobytes()


def build_difference_map_uri(ref_path, test_path):
    """Returns a data URI, or None if the map can't be produced."""
    try:
        sig_a = build_signature_mask(ref_path)
        sig_b = build_signature_mask(test_path)
        if sig_a is None or sig_b is None:
            return None
        png = render_difference_map(sig_a, sig_b)
        return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')
    except Exception:
        app.logger.exception('Difference map failed')
        return None


# ---------------------------------------------------------
# 5. ROUTES
# ---------------------------------------------------------
def render_error(message, status):
    return render_template('error.html', message=message), status


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return render_error(
        f"That upload is too large. Each image can be up to "
        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)


@app.after_request
def add_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    if request.path == '/verify':
        # Result pages contain the user's signatures - don't let anything cache them.
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/verify', methods=['POST'])
def verify():
    if not (MODEL_READY or ALLOW_UNTRAINED):
        return render_error(
            "The verification model isn't loaded on this server, so no "
            "comparison can be made.", 503)

    try:
        ref_img = read_signature('ref', 'reference')
        test_img = read_signature('test', 'test')
    except InputError as e:
        return render_error(str(e), 400)

    try:
        # Files exist only inside this block and are deleted on exit.
        with tempfile.TemporaryDirectory(prefix='signaverify_') as tmp:
            ref_path = os.path.join(tmp, 'ref.png')
            test_path = os.path.join(tmp, 'test.png')
            ref_img.save(ref_path, compress_level=1)
            test_img.save(test_path, compress_level=1)

            img_a = preprocess_image(ref_path)
            img_b = preprocess_image(test_path)
            if img_a is None or img_b is None:
                return render_error(
                    "We couldn't find a signature in one of the images. "
                    "Use a photo or scan where the signature is clearly visible.", 422)

            distance = predict_distance(img_a, img_b)
            heatmap_src = build_difference_map_uri(ref_path, test_path)

        ref_src = image_to_data_uri(ref_img)
        test_src = image_to_data_uri(test_img)
    except Exception:
        app.logger.exception('Verification failed')
        return render_error(
            "Something went wrong while comparing the signatures. Try again.", 500)

    threshold = THRESHOLD
    verdict = classify(distance, threshold)
    status, detail = VERDICT_TEXT[verdict]
    similarity = max(0.0, min(100.0, (1.0 - distance / (2.0 * threshold)) * 100.0))

    return render_template(
        'result.html',
        verdict=verdict,
        status=status,
        detail=detail,
        distance=round(distance, 4),
        similarity=round(similarity, 1),
        threshold=round(threshold, 4),
        lower=round(threshold * GENUINE_FACTOR, 3),
        upper=round(threshold * DIFFERENT_FACTOR, 3),
        band_genuine_pct=round(GENUINE_FACTOR / 2 * 100, 1),
        band_different_pct=round(DIFFERENT_FACTOR / 2 * 100, 1),
        untrained=not MODEL_READY,
        ref_src=ref_src,
        test_src=test_src,
        heatmap_src=heatmap_src,
    )


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=5000)