# 🐟 Tilapia vs Non-Tilapia Fish Classification using Deep Learning

This project implements a deep learning–based image classification system to accurately distinguish **Tilapia fish** from **Non-Tilapia fish** using convolutional neural networks and transfer learning.

The model is trained, evaluated, and validated using real-world fish images and demonstrates strong generalization performance with robust evaluation metrics.

---

## 📌 Project Overview

- **Problem Type:** Binary Image Classification  
- **Classes:**  
  - Tilapia  
  - Non-Tilapia  
- **Approach:** Transfer Learning with EfficientNet  
- **Framework:** TensorFlow & Keras  
- **Environment:** Google Colab  

This project is suitable for applications in aquaculture monitoring, fish species identification, and automated visual inspection systems.

---

## 🧠 Model Architecture

- Base Model: EfficientNet (pretrained on ImageNet)
- Custom Layers:
  - Global Average Pooling
  - Dropout for regularization
  - Dense Softmax output layer
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam
- Training Strategy:
  - Data augmentation
  - Class weighting to handle class imbalance
  - Fine-tuning of top layers

---

## 📊 Model Performance

The model was evaluated on a held-out test dataset using multiple metrics.

### 🔹 Key Results
- **Test Accuracy:** ~95–96%
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Overfitting Check:** Training vs Validation accuracy curves

### 🔹 Evaluation Artifacts
The following evaluation files are included in the repository:

- `classification_report.csv`  
- `sample_predictions.png`  
- `training_validation_accuracy.png`  

These results indicate strong generalization with minimal overfitting.

---

## 📁 Dataset Information

- The dataset consists of labeled fish images organized into:
  - Training set
  - Validation set
  - Test set
- The dataset is **not included** in this repository due to size constraints.
- Dataset was accessed via Google Drive during training.

The dataset follows the standard Keras directory structure.

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Google Colab  
- Git & GitHub  

---

## 🚀 How to Run the Project

1. Open the notebook in Google Colab  
2. Mount Google Drive  
3. Update dataset paths if required  
4. Run all cells sequentially  
5. View evaluation metrics and visualizations  

---

## 🎯 Applications

- Fish species identification  
- Aquaculture quality monitoring  
- Automated image-based classification systems  
- Computer vision learning projects  

---

## 🎤 Interview-Ready Summary

> This project uses transfer learning with EfficientNet to classify Tilapia vs Non-Tilapia fish images.  
> The model achieved ~96% test accuracy and was evaluated using precision, recall, F1-score, sample predictions, and training vs validation curves to ensure strong generalization.

---

## 👤 Author

**Aquasa Aziz**  
GitHub: https://github.com/AquasaAziz247  

---

## 📜 License

This project is intended for educational and research purposes.
