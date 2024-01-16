import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109
MODEL_PATH = "D:\\Desktop\\hack\\Deepfake_audio.h5"  # Replace with the actual path to your saved model

model = load_model(MODEL_PATH)

def predict_audio(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

        X = np.array([mel_spectrogram])
        y_pred = model.predict(X)
        predicted_class = np.argmax(y_pred)
        return predicted_class
    except Exception as e:
        return f"Error processing file: {e}"

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.flac")])
    if file_path:
        result_label.config(text="")
        clear_button.config(state="normal")  
        classify_button.config(state="normal")  
        result_label.config(text=f"Selected File: {os.path.basename(file_path)}")
        global current_file
        current_file = file_path

def clear_result():
    result_label.config(text="")
    clear_button.config(state="disabled")  
    classify_button.config(state="disabled")  

def classify_audio():
    prediction = predict_audio(current_file)
    result_label.config(text=f"Prediction: {'Deepfake' if prediction == 1 else 'Real'}")

root = tk.Tk()
root.title("Deepfake Audio Classification")

root.geometry("500x200") 

file_button = tk.Button(root, text="Browse Audio File", command=browse_file)
file_button.pack(pady=10)

clear_button = tk.Button(root, text="Clear", command=clear_result, state="disabled")
clear_button.pack(pady=5)

classify_button = tk.Button(root, text="Classify", command=classify_audio, state="disabled")
classify_button.pack(pady=5)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
