import os
import uuid

import joblib
import librosa
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "svmModel.pkl"
SCALER_PATH = "scaler.pkl"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Change this list to match the label order used when you trained the model
EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

model = None
scaler = None
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
    print("Model classes:", getattr(model, "classes_", None))
    print("Scaler expects features:", getattr(scaler, "n_features_in_", None))
except Exception as e:
    print(f"Error loading model/scaler: {e}")


def extract_features(file_path):
    """Extracts 162 features: ZCR(1) + Chroma(12) + MFCC(20) + RMS(1) + Mel(128)."""
    audio, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("index.html", prediction="Model failed to load. Check the terminal.")

    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        features = extract_features(file_path)
        print("-" * 40)
        print("File:", file.filename)
        print("Features shape:", features.shape)
        print("First 5 features:", features[:5])
        print("Any NaN:", np.isnan(features).any())

        scaled = scaler.transform(features.reshape(1, -1))
        print("Scaled min/max:", scaled.min(), scaled.max())

        raw = model.predict(scaled)[0]
        print("RAW PREDICTION:", repr(raw))

        # Turn a class number into an emotion name; if it's already text, show it as is
        try:
            emotion = EMOTIONS[int(raw)]
        except (ValueError, TypeError, IndexError):
            emotion = str(raw)

        return render_template("index.html", prediction=emotion)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("index.html", prediction=f"Error: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)