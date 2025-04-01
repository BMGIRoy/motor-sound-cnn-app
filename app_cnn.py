
import streamlit as st
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import os
import tempfile

# Configuration
st.set_page_config(page_title="Motor Sound CNN Classifier", layout="centered")
st.title("🎧 Motor Sound Classifier (CNN-Based)")

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    return load_model("motor_sound_cnn_model.h5")

model = load_cnn_model()

# Audio to Spectrogram Conversion
def audio_to_spectrogram(file, size=(128, 128)):
    y, sr = librosa.load(file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buf.name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(buf.name).convert("RGB").resize(size)
    os.unlink(buf.name)
    return np.array(img) / 255.0

# UI for file upload
st.subheader("🔍 Upload a motor sound (.wav)")
uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file:
    with st.spinner("Analyzing sound..."):
        spectrogram = audio_to_spectrogram(uploaded_file)
        input_array = np.expand_dims(spectrogram, axis=0)  # add batch dimension
        prediction = model.predict(input_array)[0][0]
        label = "Abnormal" if prediction >= 0.5 else "Normal"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.success(f"🧠 Prediction: **{label}**") 
Confidence: {confidence:.2f}")
        st.image(spectrogram, caption="Generated Mel-Spectrogram", use_column_width=True)
