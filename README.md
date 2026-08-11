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

