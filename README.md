# DeepFake Detective

A multi-model deepfake detection system built with Flask and TensorFlow.

## Models
- LeNet (64×64)
- AlexNet (64×64)  
- VGG16 (224×224) — pretrained ImageNet
- ResNet50 (224×224) — pretrained ImageNet

## Dataset
Trained on [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (FFHQ + StyleGAN).

## Features
- Upload any face image for analysis
- Grad-CAM heatmaps per model
- FFT frequency spectrum
- Weighted ensemble final verdict (95% weight on VGG16 + ResNet50)

## Limitation
> Trained exclusively on StyleGAN deepfakes — results are unreliable for 
> diffusion-generated faces (DALL-E, Midjourney, Stable Diffusion), 
> face-swap deepfakes, and other manipulation methods not represented 
> in the training set.

## Setup

### 1. Clone the repo
git clone https://github.com/aryashekhar564-ai/deepfake-detective.git
cd deepfake-detective

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add model files
Download trained models and place in models/ folder:
- LeNet_best.keras
- AlexNet_best.keras
- VGG16_best.keras
- ResNet50_best.keras

### 4. Run
python df_app.py

Open http://localhost:5000

## Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| LeNet | ~50% | ~49% | ~50% |
| AlexNet | ~52% | ~51% | ~52% |
| VGG16 | ~94% | ~94% | ~98% |
| ResNet50 | ~93% | ~93% | ~97% |

## Project Structure
deepfake_app/
├── df_app.py
├── requirements.txt
├── models/          ← not included, download separately
└── templates/
    └── index.html

## Notebooks
- deepfake_cnn_ensemble_training.ipynb — training pipeline
- deepfake_spectral_gradcam_analysis.ipynb — FFT and Grad-CAM analysis
