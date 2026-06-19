# 🩺 Detecting Pneumonia from Chest X-Ray Images Using Deep Learning

This project leverages Deep Learning and Transfer Learning techniques to automatically detect pneumonia from chest X-ray images. A custom CNN, ResNet50V2, and ResNet101V2, are trained and evaluated to compare their effectiveness in binary image classification.

---

## 📌 Project Overview

Pneumonia is a lung infection that can be diagnosed through chest X-ray imaging. Manual interpretation of medical images can be time-consuming and prone to error. This project explores the use of Convolutional Neural Networks (CNNs) and pretrained ResNet architectures to assist in the automated detection of pneumonia.

### Models Implemented
- Custom CNN
- ResNet50V2 (Transfer Learning)
- ResNet101V2 (Transfer Learning)

---

## 📂 Dataset

The project uses the Chest X-Ray Pneumonia dataset containing two classes:

- NORMAL
- PNEUMONIA

Dataset Structure:

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Transfer Learning
- ResNet50V2
- ResNet101V2

---

## 🔍 Data Preprocessing

The preprocessing pipeline includes:

- Image resizing (256 × 256)
- Pixel normalization
- Data augmentation
- Batch processing

### Data Augmentation

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])
```

---

## 🧠 Model Architectures

### 1. Custom CNN

Features:

- Convolutional Layers
- Max Pooling Layers
- Batch Normalization
- Dropout Regularization
- Dense Classification Layers

### 2. ResNet50V2

Transfer learning model using ImageNet pretrained weights.

Features:

- Frozen convolutional base
- Global Average Pooling
- Dense classifier head
- Dropout regularization

### 3. ResNet101V2

A deeper residual network pretrained on ImageNet.

Features:

- Advanced feature extraction
- Better representation learning
- Custom classification head

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Image Size | 256 × 256 |
| Batch Size | 128 |
| Optimizer | Adam |
| CNN Learning Rate | 0.0005 |
| ResNet Learning Rate | 0.0001 |
| Epochs | 10 |
| Early Stopping | Patience = 3 |

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Loss
- Validation Loss

Example:

```python
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

## 📊 Results

The notebook compares the performance of:

- Custom CNN
- ResNet50V2
- ResNet101V2

Performance is visualized using:

- Accuracy Curves
- Loss Curves
- Test Set Evaluation Metrics

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/Detecting-Pneumonia.git
cd Detecting-Pneumonia
```

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

### Launch Jupyter Notebook

```bash
jupyter notebook Detecting_Pneumonia.ipynb
```

---

## 📁 Project Structure

```text
Detecting-Pneumonia/
│
├── Detecting_Pneumonia.ipynb
├── README.md
└── chest_xray/
```

---

## 🎯 Future Enhancements

- Fine-tune pretrained ResNet layers
- Experiment with EfficientNet architectures
- Add Grad-CAM visualizations for explainability
- Hyperparameter optimization
- Deploy as a web application

---

## 📚 References

- TensorFlow Documentation
- Keras Transfer Learning Guide
- Chest X-Ray Pneumonia Dataset
- ResNet: Deep Residual Learning for Image Recognition
