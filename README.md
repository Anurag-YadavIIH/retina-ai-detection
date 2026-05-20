# Retinal Disease Detection AI

An end-to-end healthcare AI web application that detects diabetic retinopathy
from retinal fundus images using deep learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Live Demo

[retina-ai-detection.onrender.com](https://retina-ai-detection.onrender.com)

---

## Overview

Diabetic retinopathy (DR) is a leading cause of blindness worldwide. Early
detection is critical. This project uses transfer learning with ResNet18 to
classify retinal images into five severity levels and generates Grad-CAM
heatmaps to explain model predictions — making the AI interpretable for
clinical use.

## Features

- Upload retinal fundus images via drag-and-drop web interface
- Five-class severity classification: No DR / Mild / Moderate / Severe / Proliferative
- Confidence score with visual probability bars for all classes
- Grad-CAM heatmap overlay showing which retinal regions influenced the prediction
- Medical disclaimer and responsible AI design

## Architecture

```
User uploads image
      │
      ▼
Flask backend (app.py)
      │
      ▼
Preprocessing pipeline (utils/preprocess.py)
  - Resize to 224×224
  - ImageNet normalisation
      │
      ▼
ResNet18 (transfer learning, fine-tuned final layer)
      │
      ├── Softmax → class probabilities + confidence
      │
      └── Grad-CAM → heatmap overlay
            │
            ▼
     Results rendered in browser
```

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning | PyTorch 2.1, torchvision |
| Model | ResNet18 (transfer learning, ImageNet) |
| Explainability | Grad-CAM |
| Web framework | Flask 3.0 |
| Image processing | OpenCV, Pillow |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Dataset | Kaggle Diabetic Retinopathy (3662 images) |
| Deployment | Render (free tier) |

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | ~71% |
| Training epochs | 10 |
| Model size | ~45 MB |
| Inference time | ~0.3s (CPU) |

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/retina-ai-detection.git
cd retina-ai-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (requires Kaggle API key)
kaggle datasets download -d sachinkumar413/diabetic-retinopathy-dataset
unzip diabetic-retinopathy-dataset.zip -d dataset/

# 5. Train the model
python train.py

# 6. Run the web app
python app.py
# Visit http://localhost:5000
```

## Dataset

Kaggle Diabetic Retinopathy Dataset by Sachin Kumar.
3662 retinal fundus images across 5 severity classes.
https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset

## Project Structure

```
retina-ai-detection/
├── app.py              Flask web server
├── train.py            Model training pipeline
├── predict.py          Inference + Grad-CAM
├── requirements.txt    Python dependencies
├── utils/
│   └── preprocess.py   Image transforms
├── templates/
│   └── index.html      Web interface
├── static/
│   └── style.css       Stylesheet
└── model/
    └── retina_model.pth  Trained weights (not in repo)
```

## Limitations and Future Work

- Dataset is relatively small (~3662 images); more data would improve accuracy
- Model is not clinically validated; not for medical use
- Future: add CLAHE preprocessing, ensemble models, patient data input form

## License

MIT License. See LICENSE file for details.

## Author

Your Name — [github.com/Anurag-YadavIIH](https://github.com/Anurag-YadavIIH/retina-ai-detection)