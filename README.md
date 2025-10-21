# 🐟 Multiclass Fish Image Classification

## Project Overview
This project focuses on classifying fish images into multiple categories using **deep learning** techniques. It combines **training a CNN from scratch** and **fine-tuning pre-trained models** to achieve high accuracy. A **Streamlit web application** is deployed for real-time predictions, allowing users to upload images and get interactive results.

---

## Skills & Technologies Learned
- **Deep Learning:** Convolutional Neural Networks (CNN), Transfer Learning  
- **Python Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, OpenCV, PIL  
- **Data Preprocessing & Augmentation:** Image rescaling, rotation, zoom, horizontal flip  
- **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- **Visualization:** Comparison of metrics via bar charts, interactive confidence plots  
- **Deployment:** Streamlit interactive web application  
- **Version Control:** GitHub repository management  

---

## Domain
**Image Classification**  

---

## Problem Statement
The goal is to **classify fish images into multiple species**. This involves:

- Training a CNN from scratch  
- Leveraging **transfer learning** with pre-trained models like VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0  
- Saving the best-performing model for later use  
- Deploying an **interactive web app** for real-time classification  

---

## Business Use Cases
- **Enhanced Accuracy:** Identify the best model architecture for fish image classification  
- **Deployment Ready:** User-friendly web application for real-time fish classification  
- **Model Comparison:** Evaluate multiple models using performance metrics to select the best  

---

## Dataset
The dataset consists of fish images categorized by species in separate folders.  

- **Training Data:** Images of multiple fish species  
- **Validation Data:** Images reserved for testing and evaluating model performance  
- **Preprocessing:** Images are rescaled to `[0,1]` and augmented for robustness  

---

## Approach

### 1. Data Preprocessing & Augmentation
- Rescale images to `[0,1]`  
- Apply **rotation**, **zoom**, **horizontal flip**  
- Load images using **ImageDataGenerator**  

### 2. Model Training
- Train a **CNN model from scratch**  
- Fine-tune **pre-trained models**:  
  - VGG16  
  - ResNet50  
  - MobileNet  
  - InceptionV3  
  - EfficientNetB0  
- Save the best-performing model for later use  

### 3. Model Evaluation
- Compare models using metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  
- Visualize model performance using **bar charts**  

### 4. Deployment
- Build a **Streamlit app** to:  
  - Upload fish images  
  - Predict fish species  
  - Display confidence scores in **interactive charts** and **color-coded tables**  

---

## Project Deliverables
- **Trained Models:** CNN and fine-tuned pre-trained models saved in `.h5` or `.keras` format  
- **Streamlit Application:** Interactive web app for real-time prediction  
- **Python Scripts:** For training, evaluation, and deployment  
- **Comparison Report:** Metrics and insights for all models  
- **GitHub Repository:** Well-documented and organized codebase  

---


