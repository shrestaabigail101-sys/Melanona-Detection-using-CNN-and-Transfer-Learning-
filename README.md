# 🩺 Melanoma Detection using CNN and Transfer Learning

A deep learning system that classifies dermoscopic/clinical skin lesion images as **Melanoma**
or **Benign**, comparing five transfer-learning CNN architectures trained on the MIDAS
dermoscopy dataset. The best-performing model is deployed as an interactive Streamlit web app.

## 🚀 Features
- Trains and compares **5 CNN architectures** via transfer learning:
  - EfficientNet
  - VGG16
  - ResNet50
  - DenseNet121
  - InceptionV3
- Upload a skin lesion image through a web interface and get an instant prediction
- Confidence score + probability breakdown for both classes
- Clinically-worded recommendations based on the prediction (e.g. "consult a dermatologist")
- Uses the **ABCDE rule** guidance for skin self-examination in the benign case

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| Model architectures | EfficientNet, VGG16, ResNet50, DenseNet121, InceptionV3 (Keras/TensorFlow) |
| Dataset | MIDAS dermoscopy dataset |
| Deployment | Streamlit |
| Image handling | Pillow (PIL), NumPy |

## 📂 Project Structure
```
├── app.py                      # Streamlit web app (loads trained VGG16 model for inference)
├── model_1_efficientnet.py     # EfficientNet training script
├── model_2_vgg.py              # VGG16 training script
├── model_3_resnet.py           # ResNet50 training script
├── model_4_densenet.py         # DenseNet121 training script
├── model_5_inception.py        # InceptionV3 training script
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Setup

1. Clone the repo
   ```bash
   git clone https://github.com/shrestaabigail101-sys/Melanona-Detection-using-CNN-and-Transfer-Learning-.git
   cd Melanona-Detection-using-CNN-and-Transfer-Learning-
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. **Model weights**: `app.py` expects a trained model file named `melanoma_vgg16_model.h5` in
   the project root. This file is not included in the repo (too large for GitHub). Train it
   yourself using `model_2_vgg.py`, or download a pre-trained copy from:
   `<add your Google Drive / Hugging Face link here>`

4. Run the app
   ```bash
   streamlit run app.py
   ```

5. Upload a skin lesion image in the browser and view the prediction.

## 📊 Model Training
Each `model_*.py` script fine-tunes a pretrained CNN backbone (ImageNet weights) on the MIDAS
dermoscopy dataset for binary classification (melanoma vs. benign). Run the corresponding
script to reproduce training for that architecture.

## ⚠️ Disclaimer
This tool is built for educational/research purposes only. It is **not a diagnostic device**
and should never replace professional medical evaluation. Predictions are not a substitute for
a dermatologist's assessment or biopsy.

## 📌 Notes
- Built as a college project comparing transfer-learning architectures for medical image
  classification.
- Image input is resized to 224×224 and normalized before inference.

## 📄 License
MIT
