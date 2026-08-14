import os
import cv2
import random
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from model import build_siamese_network, contrastive_loss

# ---------------------------------------------------------
# 1. IDENTICAL PREPROCESSING TO APP.PY
# ---------------------------------------------------------
def read_image_exif_safe(path):
    try:
        pil_img = Image.open(path)
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
    except Exception as e:
        return None

def load_and_standardize(path):
    img = read_image_exif_safe(path)
    if img is None: return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        binary = np.ones(alpha.shape, dtype=np.uint8) * 255
        binary[alpha > 0] = 0
        return binary

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if np.median(gray) < 127: gray = cv2.bitwise_not(gray)

    bg_map = cv2.GaussianBlur(gray, (75, 75), 0)
    bg_map = np.clip(bg_map, 1, 255)
    normalized = cv2.divide(gray, bg_map, scale=255)

    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def crop_to_ink(img):
    coords = cv2.findNonZero(img)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    if w > 10 and h > 10: return img[y:y+h, x:x+w]
    return img

def preprocess_image(path):
    img = load_and_standardize(path)
    if img is None: return None

    img = cv2.bitwise_not(img)
    border = 10
    img[:border, :] = 0
    img[-border:, :] = 0
    img[:, :border] = 0
    img[:, -border:] = 0

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 25 or w < 3 or h < 3:
            cv2.drawContours(img, [cnt], -1, 0, -1)

    img = crop_to_ink(img)
    if img.size == 0: return None

    h, w = img.shape
    diff = abs(h - w)
    if h > w:
        pad_left = diff // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad_left, diff - pad_left, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        pad_top = diff // 2
        img = cv2.copyMakeBorder(img, pad_top, diff - pad_top, 0, 0, cv2.BORDER_CONSTANT, value=0)

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

# ---------------------------------------------------------
# 2. DATASET PAIRING LOGIC
# ---------------------------------------------------------
def make_pairs(dataset_path):
    pairs, labels = [], []
    gen_dir = os.path.join(dataset_path, 'genuine')
    forg_dir = os.path.join(dataset_path, 'forged')
    
    authors = {}

    print("Loading and preprocessing dataset (this may take a minute)...")
    
    if os.path.exists(gen_dir):
        for f in os.listdir(gen_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Try CEDAR format first (original_1_1.png)
                parts = f.split('_')
                author_id = parts[1] if len(parts) >= 3 else 'unknown'
                
                if author_id not in authors: authors[author_id] = {'gen': [], 'forg': []}
                
                img = preprocess_image(os.path.join(gen_dir, f))
                if img is not None: authors[author_id]['gen'].append(img)
                    
    if os.path.exists(forg_dir):
        for f in os.listdir(forg_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                parts = f.split('_')
                author_id = parts[1] if len(parts) >= 3 else 'unknown'
                
                if author_id in authors:
                    img = preprocess_image(os.path.join(forg_dir, f))
                    if img is not None: authors[author_id]['forg'].append(img)

    for author, data in authors.items():
        gens, forgs = data['gen'], data['forg']
        
        # Genuine-Genuine Matches
        for i in range(len(gens) - 1):
            pairs.append([gens[i], gens[i+1]])
            labels.append(1)
            
        # Genuine-Forged Hard Negatives
        for i in range(min(len(gens), len(forgs))):
            pairs.append([gens[i], forgs[i]])
            labels.append(0)

    temp = list(zip(pairs, labels))
    random.shuffle(temp)
    pairs, labels = zip(*temp) if temp else ([], [])
    return np.array(pairs), np.array(labels)

# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    # Make sure this points to the folder containing your CEDAR images!
    dataset_folder = "dataset" 
    
    if not os.path.exists(dataset_folder):
        print(f"Error: Folder '{dataset_folder}' not found.")
        exit()
        
    pairs, labels = make_pairs(dataset_folder)
    print(f"Generated {len(pairs)} total training pairs.")
    
    if len(pairs) < 100:
        print("WARNING: Dataset is way too small. Check your folder structure!")
        exit()
        
    img_a, img_b = pairs[:, 0], pairs[:, 1]
    
    siamese = build_siamese_network()
    
    siamese.compile(
        loss=contrastive_loss, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    print("Starting Training from scratch...")
    siamese.fit([img_a, img_b], labels, batch_size=16, epochs=30, validation_split=0.15)
    
    save_path = os.path.join('saved_model', 'siamese_model.weights.h5')
    os.makedirs('saved_model', exist_ok=True)
    siamese.save_weights(save_path)
    print(f"Training Complete! Updated weights saved to: {save_path}")