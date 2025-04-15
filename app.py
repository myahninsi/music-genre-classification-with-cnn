import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
import pickle
import pandas as pd
import plotly.express as px
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ====== PAGE SETUP ======
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")
st.title("🎵 Music Genre Classifier")
st.markdown("""
### 📚 How It Works

➡️ **Step 1**: Upload an `.mp3` or `.wav` audio file  
➡️ **Step 2**: We convert it into a **Mel spectrogram**  
➡️ **Step 3**: We then use the spectrogram to classify using a **CNN model**  
➡️ **Step 4**: You get a 🎧 **Predicted Music Genre** with confidence
""")


# ====== CONFIG ======
MODEL_PATH = "models/cnn_baseline.keras"
CLASS_INDEX_PATH = "models/class_indices.pkl"
TARGET_IMAGE_SIZE = (300, 400)  # Must match your training resolution

# ====== LOAD MODEL & CLASSES ======
@st.cache_resource
def load_model_and_classes():
    model = load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "rb") as f:
        class_indices = pickle.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model_and_classes()

# ====== SIDEBAR INFO ======
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("This model converts your audio into a **Mel spectrogram** and classifies it using a Convolutional Neural Network.")
    st.markdown("**Supported Genres:**")
    for genre in sorted(idx_to_class.values()):
        st.markdown(f"- {genre.capitalize()}")

# ====== FILE UPLOAD ======
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file:
    with st.spinner("⏳ Processing and predicting..."):

        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=uploaded_file.name[-4:], delete=False) as tmp_audio:
            tmp_audio.write(uploaded_file.read())
            audio_path = tmp_audio.name

        # Convert to WAV if needed
        if uploaded_file.name.endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)
            wav_path = audio_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
        else:
            wav_path = audio_path

        # Convert to Mel spectrogram & save as image
        y, sr = librosa.load(wav_path, duration=30)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        tmp_img_path = wav_path.replace(".wav", ".png")
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        librosa.display.specshow(S_dB, sr=sr, cmap='magma', ax=ax)
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(tmp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Predict from image
        img = image.load_img(tmp_img_path, target_size=TARGET_IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        preds = model.predict(x)
        predicted_class = idx_to_class[np.argmax(preds)]
        confidence = np.max(preds)

        # Display detailed prediction chart
        with st.expander("📊 Show detailed prediction chart"):

            top_n = 3
            top_indices = preds[0].argsort()[-top_n:][::-1]
            top_classes = [idx_to_class[i] for i in top_indices]
            top_scores = [preds[0][i] for i in top_indices]

            df = pd.DataFrame({
                'Genre': top_classes,
                'Confidence': [round(score * 100, 2) for score in top_scores]
            })

            fig = px.bar(
                df, x='Confidence', y='Genre',
                orientation='h', text='Confidence',
                color='Genre', color_discrete_sequence=px.colors.sequential.Magma_r
            )
            fig.update_layout(
                showlegend=False,
                xaxis=dict(title="Confidence (%)"),
                title="Top 3 Genre Predictions"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Display results
        st.success(f"🎧 **Predicted Genre:** {predicted_class.title()}")
        st.write(f"🔍 **Confidence:** `{confidence:.2%}`")
        st.audio(uploaded_file)

        # Display the waveform plot
        # Display detailed prediction chart
        with st.expander("🖼️ Show detailed waveform plot"):
            fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
            librosa.display.waveshow(y, sr=sr, ax=ax, color="orange")
            ax.set_title("Audio Waveform", fontsize=10)
            ax.set_yticks([])
            ax.set_xticks([])
            st.pyplot(fig)

        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)

        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

# ====== FOOTER ======
st.markdown("---")
st.caption("🎶 Built with Streamlit, TensorFlow, and Librosa")
st.caption("📊 Model trained on the GTZAN dataset")