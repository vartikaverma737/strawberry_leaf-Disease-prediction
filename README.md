# Strawberry Plant Disease Detection using Deep Learning and IoT

A smart agriculture system for **early detection of strawberry plant diseases** using **deep learning (MobileNetV2, VGG-16, ResNet50)**, deployed via **Flask API** and integrated with **ThingsBoard IoT platform** for real-time monitoring.

---

## Project Overview

This project detects **diseases in strawberry leaves** using image classification models and visualizes predictions on the **ThingsBoard IoT dashboard**. The goal is to support farmers and researchers with early warning systems to improve crop health and yield.

---

## Models Used

- **VGG-16**

Trained on a custom image dataset of strawberry leaves categorized as:
- Healthy
- Leaf Scorch
- Leaf Blight

---

## Tech Stack

| Category | Technologies |
|---------|--------------|
| **Frontend** | ThingsBoard IoT Dashboard |
| **Backend** | Flask (Python REST API) |
| **ML Models** | TensorFlow / Keras |
| **Deployment** | Localhost |
| **Integration** | HTTP POST for ThingsBoard |
| **Data Storage** | Local folders |

---

## Features

- Deep learning-based leaf disease classification
- REST API for image upload and prediction
- Real-time IoT integration with ThingsBoard
- Confidence score and disease name displayed in dashboard
- Scheduled image upload from camera (24-hour cycle)


