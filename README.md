# Customer Churn Prediction Web App

This is an interactive web application built with **Streamlit** that predicts the probability of a customer churning (leaving a service) based on various input parameters. The underlying model is trained using **TensorFlow** and **scikit-learn**.

---

## 🚀 Features

- User-friendly web interface using Streamlit
- Predicts customer churn probability using a trained deep learning model
- Encodes categorical features using pre-fitted encoders (`LabelEncoder`, `OneHotEncoder`)
- Scales inputs using a saved `StandardScaler`
- Real-time prediction and result display

---

## 📁 Files in the Project

- `app.py` — Main Streamlit application
- `model.h5` — Trained Keras model file
- `OHE_geo.pkl` — OneHotEncoder for the `Geography` feature
- `label_encoder_gender.pkl` — LabelEncoder for the `Gender` feature
- `scaler.pkl` — StandardScaler used for feature scaling
- `README.md` — This file

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt