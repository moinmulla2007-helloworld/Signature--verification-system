# SignaVerify: Signature Verification System

**SignaVerify** is a Flask-based web application that compares a reference signature with a test signature and calculates their similarity using a **Siamese Neural Network**.

The system is designed to handle real-world signature images, including smartphone photographs with **shadows, uneven lighting, rotation, and orientation issues**.

> **Disclaimer:** This project is an automated similarity check designed for preliminary screening. It is not a certified legal or forensic determination of authenticity.

---

## ✨ Features

### 🖊️ Upload or Draw Signatures

* Upload reference and test signatures as **PNG/JPG** images.
* Maximum upload size: **5 MB per image**.
* Draw signatures directly using an interactive canvas.
* Supports mouse, touch, and pointer input.

### 🔍 Three-State Verification

Instead of forcing every comparison into a simple yes/no result, SignaVerify provides three possible outcomes:

* 🟢 **Likely Genuine**
* 🟡 **Inconclusive**
* 🔴 **Likely Different**

Comparisons close to the decision threshold are classified as **Inconclusive**.

### 🗺️ Difference Map

The application aligns the two signatures and generates a visual difference map.

| Color     | Meaning                                    |
| --------- | ------------------------------------------ |
| Dark Grey | Strokes present in both signatures         |
| Red       | Strokes present only in the reference      |
| Blue      | Strokes present only in the test signature |

Small positional offsets are tolerated during comparison.

> The difference map is a pixel-level comparison after alignment. It does **not** represent the regions attended to by the neural network.

### 📱 Smartphone Image Support

The preprocessing pipeline is designed to handle common problems with smartphone photographs:

* EXIF orientation
* Rotated images
* Uneven lighting
* Shadows
* Background variations
* Camera-based illumination gradients

### 💡 Illumination Normalization

Gaussian blur division is used to reduce lighting gradients and shadows.

This helps reduce the difference between:

* Scanned signatures
* Smartphone photographs
* Images captured under uneven lighting

### 🔐 Privacy-Focused Processing

Uploaded signature images are processed in a temporary directory.

* Images are deleted after the comparison finishes.
* Uploaded images are not stored in `static/`.
* Result pages use `Cache-Control: no-store`.
* Images are embedded directly into the result page instead of being exposed through persistent image URLs.

---

## 🛠️ Tech Stack

| Component           | Technology                           |
| ------------------- | ------------------------------------ |
| Backend             | Python, Flask                        |
| Template Engine     | Jinja2                               |
| Deep Learning       | TensorFlow / Keras                   |
| Neural Network      | Siamese Convolutional Neural Network |
| Computer Vision     | OpenCV                               |
| Image Processing    | Pillow                               |
| Numerical Computing | NumPy                                |
| Frontend            | HTML, CSS, JavaScript                |
| Drawing             | HTML Canvas / Pointer Events         |

---

## 🧠 How the Model Works

SignaVerify uses a **Siamese Neural Network** to compare two signature images.

Instead of directly classifying an image as genuine or forged, the network converts each signature into a numerical representation called an **embedding**.

The embeddings are then compared using Euclidean distance.

### Basic Workflow

```text
Reference Signature
        │
        ▼
   Preprocessing
        │
        ▼
Siamese Neural Network
        │
        ▼
   256-D Embedding
        │
        ├──────────────┐
        │              │
        ▼              ▼
 Test Signature    Reference
        │
        ▼
   256-D Embedding
        │
        └──────┬───────┘
               ▼
       Euclidean Distance
               │
               ▼
        Decision Threshold
               │
               ▼
   ┌───────────┼───────────┐
   ▼           ▼           ▼
Likely      Inconclusive   Likely
Genuine                     Different
```

---

## 🧬 Network Architecture

The base network consists of three convolutional blocks.

Each block contains:

```text
Conv2D
   ↓
BatchNormalization
   ↓
LeakyReLU
   ↓
MaxPooling2D
```

The resulting feature representation is flattened and passed through:

```text
Flatten
   ↓
Dense(256)
   ↓
L2 Normalization
```

The final embeddings are located on a unit hypersphere, allowing them to be compared using Euclidean distance.

### Why a Linear Dense Layer?

The `Dense(256)` embedding layer does not use ReLU activation.

This avoids unnecessarily restricting embeddings to only positive values and helps prevent **orthant trapping**.

---

## 📚 Training

The Siamese network is trained using the **CEDAR Signature Dataset**, containing genuine and forged signatures.

The model uses **contrastive loss** with a margin of `1.0`.

The objective is to:

* Reduce the distance between genuine signature pairs.
* Increase the distance between different/forged signature pairs.

---

## 📏 Similarity Measurement

After preprocessing, both signatures are passed through the same network.

The application calculates:

```text
Euclidean Distance =
distance(reference_embedding, test_embedding)
```

### Interpretation

A **smaller distance** indicates greater similarity.

A **larger distance** indicates greater difference.

---

## ⚖️ Verification Threshold

The decision threshold is loaded from:

```text
saved_model/threshold.json
```

For example:

```json
{
    "threshold": 0.30
}
```

The application uses the threshold `T` to create a three-state decision.

| Distance                 | Verdict              |
| ------------------------ | -------------------- |
| `< 0.85 × T`             | **Likely Genuine**   |
| `0.85 × T` to `1.15 × T` | **Inconclusive**     |
| `> 1.15 × T`             | **Likely Different** |

This creates a buffer around the threshold so that borderline comparisons are not forced into a binary decision.

### Similarity Percentage

The similarity percentage displayed by the application is scaled around the decision threshold.

> **The displayed percentage is not a probability of authenticity.**

---

## 📊 Evaluation

Evaluation should be performed on writers/signatures that were **not used during training**.

| Metric                     | Value         |
| -------------------------- | ------------- |
| Dataset / Split            | *To be added* |
| AUC                        | *To be added* |
| EER                        | *To be added* |
| FAR at chosen threshold    | *To be added* |
| FRR at chosen threshold    | *To be added* |
| Chosen threshold           | *To be added* |
| Threshold selection method | *To be added* |

For a more realistic evaluation, results should also be tested on datasets different from CEDAR, such as:

* BHSig260
* GPDS

Testing with **real smartphone photographs** is also recommended because results obtained exclusively from CEDAR may not represent real-world performance.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/moinmulla2007-helloworld/Signature--verification-system.git
cd Signature--verification-system
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Weights

The application requires trained model weights at:

```text
saved_model/siamese_model.weights.h5
```

The application will not provide meaningful verification results without trained weights.

If you need to train the model yourself, the expected workflow is:

```bash
python download_dataset.py
python build_dataset.py
python train_siamese.py
```

After training, make sure the generated weights are saved in:

```text
saved_model/siamese_model.weights.h5
```

> **Important:** The weights must correspond to the architecture defined in the current `model.py`. Weights generated using an older architecture may fail to load.

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The application should then be available at:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

---

## 🧪 Development Mode

For Flask auto-reload during development:

### Windows PowerShell

```powershell
$env:FLASK_DEBUG="1"
python app.py
```

### macOS / Linux

```bash
export FLASK_DEBUG=1
python app.py
```

> **Warning:** Never enable Flask debug mode on a publicly accessible server.

---

## 🌐 Production Deployment

For production deployment, use a production WSGI server such as Gunicorn.

```bash
gunicorn app:app
```

For a public deployment, HTTPS should also be configured to protect uploaded signature images during transmission.

---

## ⚙️ Configuration

| Setting               | Configuration                                           | Default |
| --------------------- | ------------------------------------------------------- | ------- |
| Decision Threshold    | `saved_model/threshold.json` or `SIGNAVERIFY_THRESHOLD` | `0.30`  |
| Debug Mode            | `FLASK_DEBUG=1`                                         | Off     |
| Allow Untrained Model | `SIGNAVERIFY_ALLOW_UNTRAINED=1`                         | Off     |

### Decision Threshold

The threshold can be specified in:

```text
saved_model/threshold.json
```

Example:

```json
{
    "threshold": 0.30
}
```

It can also be overridden using:

```bash
SIGNAVERIFY_THRESHOLD=0.30
```

Environment variables take priority over the value in `threshold.json`.

---

## ⚠️ Running Without Trained Weights

For pipeline testing only, the application can optionally be configured to run without trained weights:

```bash
SIGNAVERIFY_ALLOW_UNTRAINED=1
```

> Scores generated without properly trained weights are **meaningless and should not be interpreted as verification results**.

---

## 📁 Project Structure

```text
Signature--verification-system/
│
├── app.py
├── model.py
├── preprocessing.py
├── train_siamese.py
├── download_dataset.py
├── build_dataset.py
├── merge_datasets.py
├── test.py
├── requirements.txt
│
├── saved_model/
│   ├── siamese_model.weights.h5
│   └── threshold.json
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── error.html
│
└── static/
    └── css/
        └── style.css
```

### Main Files

| File                  | Purpose                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `app.py`              | Flask application, validation, inference and difference-map generation |
| `model.py`            | Siamese neural network architecture                                    |
| `preprocessing.py`    | Image loading, cleaning, normalization and centering                   |
| `train_siamese.py`    | Model training                                                         |
| `download_dataset.py` | Dataset downloading                                                    |
| `build_dataset.py`    | Dataset preparation                                                    |
| `merge_datasets.py`   | Dataset merging                                                        |
| `test.py`             | Testing utilities                                                      |
| `saved_model/`        | Trained weights and threshold configuration                            |
| `templates/`          | Flask/Jinja2 HTML templates                                            |
| `static/css/`         | Application styling                                                    |

---

## 🔄 Image Processing Pipeline

```text
Uploaded Image
      │
      ▼
EXIF Orientation Correction
      │
      ▼
Image Validation
      │
      ▼
Grayscale Conversion
      │
      ▼
Illumination Normalization
      │
      ▼
Noise / Background Processing
      │
      ▼
Signature Extraction
      │
      ▼
Centering & Alignment
      │
      ▼
Neural Network
```

---

## ⚠️ Limitations

### Static Images Only

The system analyzes images rather than dynamic handwriting data.

Therefore, it cannot evaluate:

* Writing speed
* Pen pressure
* Pen lifts
* Stroke order
* Writing dynamics

The comparison is primarily based on signature **shape and layout**.

### Skilled Forgeries

Carefully imitated signatures can be difficult to distinguish from genuine signatures.

Performance may decrease when dealing with highly skilled forgeries.

### Signature Variation

A person's signature can naturally vary over time.

Using only one reference signature may therefore provide an incomplete representation of that person's normal signature variation.

### Image Quality

Poor-quality input can reduce reliability.

Examples include:

* Heavy blur
* Severe cropping
* Extreme shadows
* Very low contrast
* Excessive background noise
* Extremely blank images

### Supported Formats

The application currently supports:

```text
PNG
JPG / JPEG
```

HEIC images from some smartphones must be converted before uploading.

---

## 🔐 Privacy & Security

Signature images are processed only for the requested comparison.

During processing:

1. Uploaded files are placed in a temporary directory.
2. The signatures are processed.
3. The comparison result is generated.
4. Temporary files are deleted when processing finishes.

The application does not intentionally store uploaded signature images in the project's `static/` directory.

Result pages use:

```http
Cache-Control: no-store
```

to reduce browser/proxy caching of sensitive results.

### Production Recommendations

If deploying publicly:

* Use HTTPS.
* Keep Flask debug mode disabled.
* Configure appropriate request limits.
* Use a production WSGI server.
* Protect server logs from exposing sensitive information.
* Review your deployment's temporary-file handling.
* Avoid storing uploaded signatures unless there is a clear, documented need.

---

## 🛡️ Disclaimer

> **Disclaimer:** This project is an automated similarity check designed for preliminary screening. It is not a certified legal or forensic determination of authenticity.

The results generated by SignaVerify should not be treated as definitive proof that a signature is genuine or forged.

---

## 🖼️ Screenshots

To add screenshots, create a `docs/` directory:

```text
docs/
├── home.png
└── result.png
```

Then add:

```markdown
## Screenshots

### Home Page

![SignaVerify Home Page](docs/home.png)

### Verification Result

![SignaVerify Result Page](docs/result.png)
```

---

## 🧪 Testing

Run the project's test script with:

```bash
python test.py
```

For meaningful model evaluation, testing should use signatures and writers that were not included in the training data.

---

## 🔮 Future Improvements

Potential improvements include:

* Multiple-reference signature verification
* Support for additional image formats such as HEIC
* Better mobile-camera preprocessing
* Calibration of similarity scores
* Cross-dataset evaluation
* More comprehensive automated testing
* User authentication
* Secure deployment configuration
* Improved model architectures
* Larger and more diverse training datasets
* Detailed evaluation dashboards

---

## 👨‍💻 Author

**Moin Mulla**

GitHub:
https://github.com/moinmulla2007-helloworld

---

## ⭐ Project Summary

**SignaVerify** combines computer vision, image preprocessing, and Siamese neural networks to provide an automated signature similarity screening system.

The application is designed with practical image-processing challenges in mind, including smartphone photographs, lighting variations, EXIF orientation, alignment differences, and natural uncertainty around the decision threshold.

> **SignaVerify compares signatures; it does not certify authenticity.**
