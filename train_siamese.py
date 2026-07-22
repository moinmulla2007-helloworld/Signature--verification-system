import os
import cv2
import numpy as np
import random
import tensorflow as tf
from model import build_siamese_network, contrastive_loss

IMG_SIZE = (128, 128)

def ensure_dataset_exists():
    """Builds diverse dataset pairs if dataset folders are empty."""
    genuine_dir = os.path.join("dataset", "genuine")
    forged_dir = os.path.join("dataset", "forged")
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)

    gen_files = [f for f in os.listdir(genuine_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    forg_files = [f for f in os.listdir(forged_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(gen_files) < 10 or len(forg_files) < 10:
        print("Dataset folders empty or insufficient. Generating diverse synthetic dataset...")
        
        fonts_genuine = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
        fonts_forged = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]

        # Generate 100 Genuine Signatures (John Doe style with variations)
        for i in range(100):
            img = np.ones((200, 400), dtype=np.uint8) * 255
            font = random.choice(fonts_genuine)
            thickness = random.randint(3, 5)
            angle = random.randint(10, 20)
            
            cv2.ellipse(img, (150 + random.randint(-5, 5), 100 + random.randint(-5, 5)), 
                        (80 + random.randint(-5, 5), 30 + random.randint(-5, 5)), angle, 0, 360, (0,), thickness)
            cv2.line(img, (70 + random.randint(-10, 10), 130 + random.randint(-10, 10)), 
                     (330 + random.randint(-10, 10), 70 + random.randint(-10, 10)), (0,), thickness + 1)
            cv2.putText(img, "John Doe", (90 + random.randint(-5, 5), 115 + random.randint(-5, 5)), font, 1.7, (0,), thickness)
            
            cv2.imwrite(os.path.join(genuine_dir, f"gen_{i:03d}.png"), img)

        # Generate 100 Forged Signatures (Structurally different stroke styles)
        for i in range(100):
            img = np.ones((200, 400), dtype=np.uint8) * 255
            font = random.choice(fonts_forged)
            thickness = random.randint(1, 2)
            angle = random.randint(-30, -10)
            
            cv2.ellipse(img, (200 + random.randint(-10, 10), 90 + random.randint(-10, 10)), 
                        (50 + random.randint(-5, 5), 50 + random.randint(-5, 5)), angle, 0, 360, (0,), thickness)
            cv2.line(img, (40 + random.randint(-10, 10), 160 + random.randint(-10, 10)), 
                     (360 + random.randint(-10, 10), 50 + random.randint(-10, 10)), (0,), thickness)
            cv2.putText(img, "John Doe", (70 + random.randint(-10, 10), 130 + random.randint(-10, 10)), font, 1.3, (0,), thickness)
            
            cv2.imwrite(os.path.join(forged_dir, f"forg_{i:03d}.png"), img)

        print("Dataset ready with 100 Genuine and 100 Forged samples!")

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def create_pairs(genuine_dir, forged_dir):
    gen_files = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    forg_files = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    gen_imgs = [preprocess_image(f) for f in gen_files if preprocess_image(f) is not None]
    forg_imgs = [preprocess_image(f) for f in forg_files if preprocess_image(f) is not None]
    
    pairs_a, pairs_b, labels = [], [], []
    
    # 1. Positive Pairs (Genuine vs Genuine -> Target Label = 1.0)
    for i in range(len(gen_imgs) - 1):
        pairs_a.append(gen_imgs[i])
        pairs_b.append(gen_imgs[i+1])
        labels.append(1.0)
        
    # 2. Negative Pairs (Genuine vs Forged -> Target Label = 0.0)
    for i in range(min(len(gen_imgs), len(forg_imgs))):
        pairs_a.append(gen_imgs[i])
        pairs_b.append(forg_imgs[i])
        labels.append(0.0)
        
    return [np.array(pairs_a), np.array(pairs_b)], np.array(labels)

if __name__ == "__main__":
    # Ensure correct target directory exists
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # Verify/populate dataset
    ensure_dataset_exists()
    
    print("Loading image pairs...")
    X, y = create_pairs("dataset/genuine", "dataset/forged")
    print(f"Total training pairs: {len(y)}")
    
    print("Building Siamese Network architecture...")
    siamese = build_siamese_network()
    siamese.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=contrastive_loss(margin=1.0))
    
    print("Training Siamese Network...")
    siamese.fit(X, y, batch_size=16, epochs=15, validation_split=0.2)
    
    # Save directly to saved_model/siamese_model.weights.h5
    weight_path = os.path.join(save_dir, "siamese_model.weights.h5")
    siamese.save_weights(weight_path)
    
    print("=" * 50)
    print(f"SUCCESS: Model weights saved to {weight_path}")
    print("=" * 50)