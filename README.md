# Melanoma Detection using CNN & Transfer Learning

A deep learning-based system for classifying skin lesion images as melanoma or non-melanoma, built by training a custom CNN alongside five pretrained transfer learning architectures — Efficientnet, VGG, Inception, ResNet and DenseNet. The best-performing model (selected using metrics like accuracy, sensitivity, and specificity) powers an interactive Streamlit web app that lets users upload a skin lesion image and receive an instant diagnosis prediction along with a confidence score that provides justification.

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Transfer Learning Models** | VGG16, InceptionV3, ResNet50, DenseNet121, EfficientNetB0 |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-learn |
| **Web App / Deployment** | Streamlit |
| **Environment** | Jupyter Notebook / Google Colab |

---

## Architecture

```
                          ┌────────────────────────┐
                          │   Skin Lesion Dataset  │
                          │   (Stanford Dataset)   │
                          └───────────┬────────────┘
                                      │
                         Preprocessing & Augmentation
                (resize, normalize, rotation, flip, zoom, etc.)
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
             Train/Val/Test Split                Class Balancing
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
      ┌───────▼──────┐       ┌────────▼──────────┐      ┌────────▼──────────┐
      │  Custom CNN  │       │ Transfer Learning │      │ Transfer Learning │
      │   (from      │       │  Models (frozen   │      │  Models (fine-    │
      │   scratch)   │       │  base + custom    │      │  tuned layers)    │
      └───────┬──────┘       │  classifier head) │      └────────┬──────────┘
              │              └─────────┬─────────┘               │
              └────────────────────────┼────────────────────────────┘
                                       │
                         Model Training & Validation
                                       │
                    Evaluation (Accuracy, Sensitivity,
                        Specificity, AUC-ROC, F1-Score)
                                       │
                         Best Model Selection
                                       │
                         Model Serialization (.h5 / .keras)
                                       │
                         ┌─────────────▼─────────────┐
                         │      Streamlit Web App    │
                         │  ┌────────────────────────│
                         │  │  Image Upload Interface│ │
                         │  ├────────────────────────┤ │
                         │  │  Preprocessing Pipeline│ │
                         │  ├────────────────────────┤ │
                         │  │   Model Inference      │ │
                         │  ├────────────────────────┤ │
                         │  │  Diagnosis Output +    │ │
                         │  │  Confidence Score      │ │
                         │  └────────────────────────┘ │
                         └─────────────────────────────┘
```

---

## Models Compared

| Model | Type | Notes |
|---|---|---|
| VGG16/19 | Transfer Learning | Deep, uniform convolutional blocks |
| InceptionV3 | Transfer Learning | Multi-scale feature extraction via inception modules |
| ResNet50 | Transfer Learning | Residual connections to combat vanishing gradients |
| DenseNet121 | Transfer Learning | Dense connectivity for feature reuse |
| EfficientNetB0 | Transfer Learning | Compound scaling for efficiency and accuracy |

---

## Evaluation Metrics

Models were compared using the following metrics to account for the clinical importance of minimizing false negatives:

- **Accuracy** — overall correctness of predictions
- **Sensitivity (Recall)** — ability to correctly identify melanoma cases (critical for early detection)
- **Specificity** — ability to correctly identify non-melanoma cases
- **Precision**
- **F1-Score**
- **AUC-ROC Curve**
- **Confusion Matrix**

The model with the best balance of sensitivity and specificity (prioritizing sensitivity, given the cost of missing a melanoma case) was selected for deployment.

---

## Future Improvements

- Incorporate Grad-CAM/explainability to visualize regions influencing the model's decision
- Expand to multi-class classification (e.g., melanoma, nevus, basal cell carcinoma, etc.)
- Add ensemble modeling across top-performing architectures
- Deploy on cloud (Streamlit Community Cloud / AWS / Azure) with a public inference endpoint
- Improve dataset diversity to reduce bias across skin tones

---

## Disclaimer

This project is intended for educational and research purposes only. It is not a certified medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for skin health concerns.
