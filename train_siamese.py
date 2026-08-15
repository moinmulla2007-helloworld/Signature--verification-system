import os
import random
import numpy as np
import tensorflow as tf
from model import build_siamese_network, contrastive_loss
from preprocessing import preprocess_for_model

# ---------------------------------------------------------
# 1. IMAGE PREPROCESSING - imported from preprocessing.py, NOT
# redefined here. This used to be its own copy of preprocess_image()
# that silently drifted out of sync with app.py's - training on that
# stale pipeline while serving inference through the updated one would
# have produced a model tuned to a different input distribution than
# what it sees in production. Both scripts must always go through
# preprocessing.preprocess_for_model() so they can never diverge again.
# ---------------------------------------------------------
def preprocess_image(path):
    img = preprocess_for_model(path, canvas_size=256)
    if img is None:
        return None
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

                if author_id not in authors:
                    authors[author_id] = {'gen': [], 'forg': []}

                img = preprocess_image(os.path.join(gen_dir, f))
                if img is not None:
                    authors[author_id]['gen'].append(img)

    if os.path.exists(forg_dir):
        for f in os.listdir(forg_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                parts = f.split('_')
                author_id = parts[1] if len(parts) >= 3 else 'unknown'

                if author_id in authors:
                    img = preprocess_image(os.path.join(forg_dir, f))
                    if img is not None:
                        authors[author_id]['forg'].append(img)

    for author, data in authors.items():
        gens, forgs = data['gen'], data['forg']

        # Genuine-Genuine Matches
        for i in range(len(gens) - 1):
            pairs.append([gens[i], gens[i + 1]])
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

    # --- Embedding collapse sanity check ---
    # Runs BEFORE the full fit() call so a collapsing model is caught in
    # seconds, not after a multi-hour CPU run. Pulls a small batch through
    # the base network only (pre-L2-normalize) and prints the spread of
    # embedding norms - these should NOT all cluster near 0.
    base_network = siamese.get_layer('base_network')
    sample_batch = img_a[:min(16, len(img_a))]
    pre_norm_model = tf.keras.Model(
        base_network.input,
        base_network.get_layer('l2_norm').input  # embedding BEFORE l2_normalize
    )
    sample_embeddings = pre_norm_model.predict(sample_batch, verbose=0)
    norms = np.linalg.norm(sample_embeddings, axis=1)
    print(f"Pre-normalization embedding norms (should be spread out, not "
          f"clustered near 0): min={norms.min():.4f} max={norms.max():.4f} "
          f"mean={norms.mean():.4f}")
    if norms.max() < 1e-3:
        print("WARNING: embeddings look collapsed before training even "
              "starts. Check model.py and this preprocessing pipeline "
              "before continuing.")

    print("Starting Training from scratch...")
    siamese.fit([img_a, img_b], labels, batch_size=16, epochs=30, validation_split=0.15)

    save_path = os.path.join('saved_model', 'siamese_model.weights.h5')
    os.makedirs('saved_model', exist_ok=True)
    siamese.save_weights(save_path)
    print(f"Training Complete! Updated weights saved to: {save_path}")