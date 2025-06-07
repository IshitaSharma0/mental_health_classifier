# Mental Health Risk Prediction App

## Description
This project is an NLP-based Streamlit web app designed to detect signs of mental health distress in user-written text using a trained deep learning model.

## 🚀 Features
- Text preprocessing with lemmatization  
- Bi-directional LSTM model architecture  
- Focal loss function to handle class imbalance  
- Custom tokenizer (`tokenizer.pkl`) for consistent text processing  
- Pretrained model (`mental_health_model.h5`) for prediction  
- Streamlit-based interactive frontend  

## 📁 Files
- `app.py` — Streamlit application script  
- `mental_health_model.h5` — Trained Keras deep learning model  
- `tokenizer.pkl` — Saved tokenizer object used for text preprocessing  
- `requirements.txt` — Python dependencies required for the project  
- `README.md` — Project overview and instructions  

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>

