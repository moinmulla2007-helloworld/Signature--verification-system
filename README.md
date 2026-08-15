# SignaVerify: Real-World Signature Verification System

An end-to-end Machine Learning web application built to detect forged signatures. Unlike standard tutorial models that only work on pristine digital datasets, this system is engineered to handle messy, real-world smartphone photos using a robust Siamese Neural Network and advanced Computer Vision preprocessing pipelines.

## 🚀 Features

* **EXIF-Safe Computer Vision:** Natively handles smartphone camera orientation tags using `PIL` to prevent sideways/flipped image processing.
* **Illumination Normalization:** Employs Gaussian blur division to erase room shadows and lighting gradients from camera photos, bridging the domain gap between scanned documents and mobile uploads.
* **Anti-Collapse Siamese Architecture:** Utilizes `LeakyReLU` activations and a linear `Dense(256)` layer followed by L2-Normalization to project feature embeddings onto a full unit hypersphere, entirely preventing the "mode collapse" common in contrastive learning.
* **Explainable AI (XAI) Heatmaps:** Generates an interpretability difference map for every comparison. Overlapping/matching strokes are highlighted in green, while divergent strokes are highlighted in red.
* **Zero-Shot Verification:** Tuned to a realistic human biomechanical variance threshold (`0.65` with a margin of `1.0`), allowing it to verify names and signatures it has never seen before.

## 🛠️ Tech Stack

* **Backend & Web Framework:** Python, Flask, Werkzeug
* **Deep Learning:** TensorFlow / Keras (Siamese Convolutional Neural Network)
* **Computer Vision:** OpenCV (`cv2`), Pillow (`PIL`)
* **Data Processing:** NumPy
* **Frontend:** HTML5, CSS3, Canvas API (for live signature drawing)

## 🧠 Model Architecture

The core of the system is a Siamese Neural Network trained on the **CEDAR Signature Dataset** (intra-author pairs for strict genuine/forged feature separation).

1. **Base Network:** 3 blocks of `Conv2D` + `BatchNormalization` + `LeakyReLU` + `MaxPooling2D`.
2. **Embedding:** Flattened into a linear `Dense(256)` layer (no ReLU, preventing orthant trapping).
3. **Normalization:** `Lambda` layer applying L2-Normalization.
4. **Loss Function:** Custom Contrastive Loss with a strict margin of `1.0`.

