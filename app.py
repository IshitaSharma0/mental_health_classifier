import streamlit as st
import numpy as np
import re
import tensorflow as tf
from keras_cv.losses import FocalLoss
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import pickle

# Download WordNet (first time only)
nltk.download('wordnet')

# ---- Load resources ----

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model(
    "mental_health_model.h5",
    custom_objects={'FocalLoss': FocalLoss()},
    compile=False  # Skip loading optimizer state to avoid error
)

# Now compile the model manually after loading
model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])


# Constants
max_length = 100
important_terms = ["depressed", "suicidal", "anxious", "bipolar", "sad", "hopeless"]
lemmatizer = WordNetLemmatizer()

# ---- Preprocessing pipeline ----

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def emphasize_keywords(text):
    for word in important_terms:
        text = text.replace(word, (word + " ") * 3)
    return text

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text_pipeline(text):
    text = clean_text(text)
    text = emphasize_keywords(text)
    text = lemmatize_text(text)
    return text

# ---- Streamlit UI ----

st.set_page_config(page_title="Mental Health Risk Classifier", layout="centered")
st.title("🧠 Mental Health Text Classifier")
st.write("Enter a comment below to assess its potential mental health risk.")

user_input = st.text_area("📝 Enter a mental health-related comment:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Preprocess and predict
        processed = preprocess_text_pipeline(user_input)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')
        prob = model.predict(padded, verbose=0)[0][0]
        label = "⚠️ High Risk" if prob >= 0.3 else "✅ Low Risk"

        st.subheader("Prediction:")
        st.markdown(f"**{label}**")
        st.write(f"**Probability:** {prob:.2f}")
