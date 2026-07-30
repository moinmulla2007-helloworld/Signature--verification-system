import os
import cv2
import numpy as np
import random
import tensorflow as tf
from itertools import combinations
from model import build_siamese_network, contrastive_loss

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
            
    if img.size == 0: return None

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

    # Resize & Normalize
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def augment_signature(img_array, num_variations=15):
    img = img_array[:, :, 0]
    aug_images = [img_array] 
    
    for _ in range(num_variations):
        angle = np.random.uniform(-12, 12)
        M_rot = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
        rotated = cv2.warpAffine(img, M_rot, (128, 128), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        tx = np.random.uniform(-8, 8)
        ty = np.random.uniform(-8, 8)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(rotated, M_trans, (128, 128), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        shifted = np.expand_dims(shifted, axis=-1)
        aug_images.append(shifted)
        
    return aug_images

def make_pairs(dataset_path):
    pairs, labels = [], []
    people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    person_data = {}
    
    for person in people:
        gen_dir = os.path.join(dataset_path, person, 'genuine')
        forg_dir = os.path.join(dataset_path, person, 'forged')
        
        gen_paths = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if os.path.isfile(os.path.join(gen_dir, f))]
        forg_paths = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir) if os.path.isfile(os.path.join(forg_dir, f))]
        
        gen_imgs = []
        for p in gen_paths:
            img = preprocess_image(p)
            if img is not None: gen_imgs.extend(augment_signature(img, 15))
                
        forg_imgs = []
        for p in forg_paths:
            img = preprocess_image(p)
            if img is not None: forg_imgs.extend(augment_signature(img, 15))
                
        person_data[person] = {'genuine': gen_imgs, 'forged': forg_imgs}

    # Match Pairs
    for person, data in person_data.items():
        gens = data['genuine']
        for i in range(len(gens) - 1):
            pairs.append([gens[i], gens[i+1]])
            labels.append(0)

    # Intra-Person Forgeries
    for person, data in person_data.items():
        gens, forgs = data['genuine'], data['forged']
        for i in range(min(len(gens), len(forgs))):
            pairs.append([gens[i], forgs[i]])
            labels.append(1)

    # Inter-Person Differences
    people_list = list(person_data.keys())
    if len(people_list) > 1:
        for p1, p2 in combinations(people_list, 2):
            g1, g2 = person_data[p1]['genuine'], person_data[p2]['genuine']
            for i in range(min(len(g1), len(g2))):
                pairs.append([g1[i], g2[i]])
                labels.append(1)

    temp = list(zip(pairs, labels))
    random.shuffle(temp)
    pairs, labels = zip(*temp)
    return np.array(pairs), np.array(labels)

if __name__ == "__main__":
    dataset_folder = "custom_dataset"
    if not os.path.exists(dataset_folder):
        print(f"Error: Folder '{dataset_folder}' not found.")
        exit()
        
    print("Generating enriched pair dataset...")
    pairs, labels = make_pairs(dataset_folder)
    print(f"Generated {len(pairs)} total training pairs.")
    
    img_a, img_b = pairs[:, 0], pairs[:, 1]
    
    siamese = build_siamese_network()
    siamese.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
    
    print("Starting Training (50 Epochs)...")
    siamese.fit([img_a, img_b], labels, batch_size=16, epochs=50, validation_split=0.15)
    
    os.makedirs('saved_model', exist_ok=True)
    save_path = 'saved_model/siamese_model.weights.h5'
    siamese.save_weights(save_path)
    print(f"Training Complete! Saved to: {save_path}")