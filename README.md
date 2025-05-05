# 🧠 Anomaly Detection in Fashion MNIST using CNN and Autoencoder

This project focuses on detecting anomalies in the Fashion MNIST dataset using two approaches: a **Convolutional Neural Network (CNN)** for classification and an **Autoencoder** for unsupervised anomaly detection. It explores how deep learning models can be used not only to classify fashion items but also to detect corrupted or unusual patterns that deviate from the norm.

> Developed by: Yash Dixit & Mahi Attri

---

## 📌 Table of Contents

1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Methodologies](#methodologies)
   - [CNN for Classification](#cnn-for-classification)
   - [Synthetic Anomaly Creation](#synthetic-anomaly-creation)
   - [Autoencoder for Anomaly Detection](#autoencoder-for-anomaly-detection)
5. [Evaluation Strategy](#evaluation-strategy)
6. [Results & Visualizations](#results--visualizations)
7. [Technologies Used](#technologies-used)
8. [Future Work](#future-work)
9. [Contributors](#contributors)
 
---

## 📖 Introduction

In the real world, identifying anomalies in image data is crucial for quality control, fraud detection, and safety assurance. Traditional classification models often fail to recognize data that deviates significantly from training examples.

In this project, we aim to:
- Train a CNN to classify fashion items.
- Introduce artificial anomalies by applying noise and transformations.
- Train an autoencoder to reconstruct normal images and use reconstruction errors to detect anomalies.

---

## 🎯 Objectives

- Build a robust **CNN model** to classify Fashion MNIST images.
- Create **synthetic anomalies** to simulate corrupted or unusual data.
- Use an **Autoencoder** trained on normal data to detect anomalies.
- Analyze the performance using visualization and evaluation metrics.

---

## 📂 Dataset Description

The **Fashion MNIST** dataset contains grayscale images of 28x28 pixels representing 10 classes of clothing, such as:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

**Data Split:**
- Training set: 60,000 images
- Test set: 10,000 images

---

## 🧪 Methodologies

### 🧱 CNN for Classification

We first build a CNN to classify the fashion items into their respective categories.

#### Key Features:
- **Input**: 28x28 grayscale images
- **Layers**:
  - 3 Convolutional layers with ReLU activation
  - MaxPooling layers to reduce spatial dimensions
  - Flattening and Dense layers
  - Dropout for regularization
- **Output**: 10-class softmax classifier

The CNN is trained on the original (normal) images to ensure it learns accurate class-specific features.

---

### 🧪 Synthetic Anomaly Creation

To simulate anomalies:
- **Gaussian Noise** is added to clean test images
- **Random Rotations** are applied to disorient the features
- A combination of both is used to create severely distorted samples

These modified images are labeled as `-1` (anomalies), while original images are labeled as `0` (normal) for evaluation.

---

### 🔄 Autoencoder for Anomaly Detection

The Autoencoder is an unsupervised neural network that learns to **reconstruct its input**. It is trained exclusively on normal data so that it cannot effectively reconstruct anomalous inputs.

#### Architecture:
- **Encoder**:
  - Flattens input
  - Dense layers to reduce dimension to 64 (latent space)
- **Decoder**:
  - Dense layers to reconstruct back to 28x28
- **Loss Function**: Binary Crossentropy

During inference, anomalous samples show **high reconstruction error**, enabling us to distinguish them.

---

## 📏 Evaluation Strategy

1. **Reconstruction Error Calculation**:
   - Compute MSE (Mean Squared Error) between original and reconstructed images

2. **Thresholding**:
   - Set threshold at the **95th percentile** of MSE values for normal data

3. **Labeling**:
   - If error > threshold → Anomaly (`1`)
   - Else → Normal (`0`)

4. **Metrics Used**:
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1)
   - ROC-AUC Score

---

## 📊 Results & Visualizations

### 🧮 Performance Highlights

- Autoencoder achieved a **ROC-AUC score** indicating effective separation between normal and anomaly.
- CNN achieved high accuracy on classification of normal images.
- Visual analysis confirms that reconstruction error is significantly higher for anomalies.

### 📸 Visualizations Included:

- Training Loss Curves
- Original vs. Reconstructed Images
- Histogram of Reconstruction Errors
- ROC Curve
- CNN Accuracy and Confusion Matrix

---

## ⚙️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras**
- **NumPy & Matplotlib**
- **Scikit-learn**
- **Google Colab** (for training and visualization)

---

## 🔮 Future Work

- Extend to **real-world anomaly detection** scenarios (e.g., medical images).
- Experiment with **Variational Autoencoders (VAE)** and **GANs** for more robust detection.
- Apply **explainability techniques** like Grad-CAM for interpreting CNN decisions.
- Integrate into an **Edge device or mobile app** using TensorFlow Lite.

---

## 👥 Contributors

- Yash Dixit
- Mahi Attri

---
 

