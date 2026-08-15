"""
Single source of truth for signature image preprocessing.

IMPORTANT: this module exists specifically to prevent train/inference
skew. Earlier in this project, app.py and train.py each had their own
copy of preprocess_image(), and app.py's copy got fixed (EXIF rotation,
torn-edge noise cleanup, centroid+height-based centering) while train.py's
copy was never updated - so a model trained on train.py's stale pipeline
was being served through app.py's much cleaner one. Both files now import
everything from here. If you need to change how images are preprocessed,
change it ONCE, in this file - never re-copy these functions into app.py
or train.py directly.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps


def read_image_exif_safe(path):
    """
    Reads an image the way a human (and a browser <img> tag) sees it.

    cv2.imread() ignores the EXIF Orientation tag entirely, so a portrait
    phone photo stored with Orientation=6/8/3 comes in still rotated on
    its side even though the frontend displays it upright. This function
    opens the file with Pillow, physically bakes the EXIF rotation into
    the pixel data with ImageOps.exif_transpose(), and only then converts
    to a numpy array in the channel order OpenCV expects.

    Returns a numpy array: grayscale (H,W), BGR (H,W,3), or BGRA (H,W,4).
    Returns None if the file can't be opened.
    """
    try:
        pil_img = Image.open(path)
    except Exception:
        return None

    pil_img = ImageOps.exif_transpose(pil_img)

    mode = pil_img.mode
    if mode == 'RGBA':
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif mode == 'RGB':
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif mode in ('L', 'LA', '1'):
        return np.array(pil_img.convert('L'))
    else:
        arr = np.array(pil_img.convert('RGB'))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if np.median(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Illumination Normalization (Shadow Removal via Gaussian division)
    bg_map = cv2.GaussianBlur(gray, (75, 75), 0)
    bg_map = np.clip(bg_map, 1, 255)
    normalized = cv2.divide(gray, bg_map, scale=255)

    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def crop_to_ink(img):
    """
    Tightly crops around all valid ink strokes. Expects ink=255, bg=0.

    Deliberately axis-aligned only (cv2.boundingRect on findNonZero).
    Do NOT replace with cv2.minAreaRect-based deskewing - its angle has
    an inherent 90/180-degree ambiguity that flips signatures. Remaining
    skew after EXIF correction is natural handwriting slant, which the
    network should learn to tolerate from training variation.
    """
    coords = cv2.findNonZero(img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    if w > 10 and h > 10:
        return img[y:y + h, x:x + w]
    return img


def clean_ink_mask(img, border_margin_ratio=0.06, min_component_area=40):
    """
    Strips non-signature noise from a binarized ink mask (ink=255, bg=0)
    BEFORE cropping: torn paper edges, staple holes, shadow speckle, etc.

    1. Proportional border wipe - a signature is never written right up
       against the physical edge of the paper, so torn edges/shadow bands
       in this margin are safe to zero out. Scales with image size rather
       than a fixed pixel count.
    2. Remove remaining small connected components (speckle noise).

    NOTE: a morphological opening step was tried here and removed - pen
    strokes are only ~2-4px wide after binarization, so erosion ate
    genuine ink and fragmented strokes into a dotted pattern that
    inflated measured distance between genuine pairs.
    """
    h, w = img.shape[:2]
    cleaned = img.copy()

    bh = max(1, int(h * border_margin_ratio))
    bw = max(1, int(w * border_margin_ratio))
    cleaned[:bh, :] = 0
    cleaned[-bh:, :] = 0
    cleaned[:, :bw] = 0
    cleaned[:, -bw:] = 0

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < min_component_area or cw < 3 or ch < 3:
            cv2.drawContours(cleaned, [cnt], -1, 0, -1)

    return cleaned


def crop_and_center(img, canvas_size=256, ink_fill_ratio=0.8):
    """
    Crops to ink, scales (aspect-preserving) so ink HEIGHT fills
    `ink_fill_ratio` of the canvas, then centers on the ink's CENTROID
    (center of mass) rather than its bounding-box center.

    Scale by height, not max(h, w): a long underline flourish inflates
    bounding-box width without making the letters bigger. Scaling by
    height keeps letterforms a consistent size regardless of flourish
    length; an overly wide flourish simply clips at the canvas edge,
    which is an acceptable tradeoff since it carries little identity
    signal compared to the letters themselves.

    Center on centroid, not bbox center: an asymmetric flourish pulls a
    bounding box off-center by a different amount per sample, so two
    genuine signatures of the same name can land with their letters in
    different spots even when both boxes are "centered." Centroid
    centering keeps the bulk of the ink anchored consistently.

    Expects ink=255 (white), background=0 (black).
    """
    cropped = crop_to_ink(img)
    if cropped.size == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    h, w = cropped.shape
    scale = (canvas_size * ink_fill_ratio) / h
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    moments = cv2.moments(resized, binaryImage=True)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = new_w / 2.0, new_h / 2.0

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    offset_x = int(round(canvas_size / 2.0 - cx))
    offset_y = int(round(canvas_size / 2.0 - cy))

    src_x0, src_y0 = max(0, -offset_x), max(0, -offset_y)
    dst_x0, dst_y0 = max(0, offset_x), max(0, offset_y)
    copy_w = min(new_w - src_x0, canvas_size - dst_x0)
    copy_h = min(new_h - src_y0, canvas_size - dst_y0)

    if copy_w > 0 and copy_h > 0:
        canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
            resized[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]

    return canvas


def preprocess_for_model(path, canvas_size=256):
    """
    The full shared pipeline: EXIF-safe read -> illumination-normalize +
    binarize -> invert to ink=255/bg=0 -> strip noise -> crop, scale by
    height, and center on centroid.

    Returns a (canvas_size, canvas_size) uint8 array (ink=255, bg=0), or
    None if the image couldn't be read. This is the exact array both
    app.py (for a live prediction) and train.py (for building training
    pairs) must use - each caller then does its own final normalization
    step (see preprocess_image() in each file), but the pixels they
    normalize must come from here, unmodified, so training and inference
    see the same distribution.
    """
    img = load_and_standardize(path)
    if img is None:
        return None

    img = cv2.bitwise_not(img)
    img = clean_ink_mask(img)
    img = crop_and_center(img, canvas_size=canvas_size, ink_fill_ratio=0.8)
    return img