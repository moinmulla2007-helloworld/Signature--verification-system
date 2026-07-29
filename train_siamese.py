import os
import cv2
import numpy as np
import random
import tensorflow as tf
from itertools import combinations
from model import build_siamese_network, contrastive_loss

def load_and_preprocess(image_path):
    """Loads and standardizes a single image with Adaptive Thresholding."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        binary = np.ones(alpha.shape, dtype=np.uint8) * 255 
        binary[alpha > 0] = 0 
    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 15
        )

    img = cv2.bitwise_not(binary)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def augment_signature(img_array, num_variations=15):
    """DATA AUGMENTATION: Artificially multiplies your dataset."""
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
    pairs = []
    labels = []
    people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    person_data = {}
    
    # Load all images per person
    for person in people:
        person_dir = os.path.join(dataset_path, person)
        gen_dir = os.path.join(person_dir, 'genuine')
        forg_dir = os.path.join(person_dir, 'forged')
        
        gen_paths = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if os.path.isfile(os.path.join(gen_dir, f))]
        forg_paths = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir) if os.path.isfile(os.path.join(forg_dir, f))]
        
        gen_imgs = []
        for p in gen_paths:
            img = load_and_preprocess(p)
            if img is not None:
                gen_imgs.extend(augment_signature(img, 15))
                
        forg_imgs = []
        for p in forg_paths:
            img = load_and_preprocess(p)
            if img is not None:
                forg_imgs.extend(augment_signature(img, 15))
                
        person_data[person] = {'genuine': gen_imgs, 'forged': forg_imgs}

    # 1. Create Positive Pairs (Genuine A + Genuine A) -> Label 0
    for person, data in person_data.items():
        gens = data['genuine']
        for i in range(len(gens) - 1):
            pairs.append([gens[i], gens[i+1]])
            labels.append(0)

    # 2. Create Intra-Person Negative Pairs (Genuine A + Forged A) -> Label 1
    for person, data in person_data.items():
        gens = data['genuine']
        forgs = data['forged']
        for i in range(min(len(gens), len(forgs))):
            pairs.append([gens[i], forgs[i]])
            labels.append(1)

    # 3. Create Inter-Person Negative Pairs (Genuine Person A + Genuine Person B) -> Label 1
    people_list = list(person_data.keys())
    if len(people_list) > 1:
        for p1, p2 in combinations(people_list, 2):
            g1 = person_data[p1]['genuine']
            g2 = person_data[p2]['genuine']
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
    
    img_a = pairs[:, 0]
    img_b = pairs[:, 1]
    
    print("Building Siamese Network...")
    siamese = build_siamese_network()
    siamese.compile(
        loss=contrastive_loss, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003)
    )
    
    print("Starting Training (50 Epochs)...")
    siamese.fit(
        [img_a, img_b], 
        labels, 
        batch_size=16, 
        epochs=50, 
        validation_split=0.15 
    )
    
    os.makedirs('saved_model', exist_ok=True)
    save_path = 'saved_model/siamese_model.weights.h5'
    siamese.save_weights(save_path)
    print(f"Training Complete! Retrained model saved to: {save_path}")