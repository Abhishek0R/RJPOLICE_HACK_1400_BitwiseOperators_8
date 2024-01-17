from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109
AUDIO_MODEL_PATH = "D:\\Desktop\\hack\\Deepfake_audio.h5"

IMAGE_DIMENSIONS = {'height': 256, 'width': 256, 'channels': 3}
IMAGE_MODEL_PATH = "D:\\Desktop\\hack\\MesoInception_F2F.h5"


class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


class MesoInception4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    @staticmethod
    def InceptionLayer(a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y

        return func

    def init_model(self):
        x = Input(shape=(IMAGE_DIMENSIONS['height'],
                         IMAGE_DIMENSIONS['width'],
                         IMAGE_DIMENSIONS['channels']))
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)


class DeepfakeClassifier:
    def __init__(self):
        self.audio_model = load_model(AUDIO_MODEL_PATH)
        self.image_model = MesoInception4()
        self.image_model.load(IMAGE_MODEL_PATH)
        self.current_audio_file = None
        self.current_image_file = None
        self.root = tk.Tk()
        self.root.title("Deepfake Classifier")
        self.root.geometry("600x300")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        logo_path = "D:\\Desktop\\hack\\logo.png"  # Replace with the actual path to your logo
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((50, 50), Image.BICUBIC)
        logo_tk_image = ImageTk.PhotoImage(logo_image)

        self.logo_label = tk.Label(self.root, image=logo_tk_image)
        self.logo_label.image = logo_tk_image
        self.logo_label.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

        self.audio_label = tk.Label(self.root, text="Choose an audio file:")
        self.audio_label.grid(row=1, column=0, pady=10, sticky="w")

        self.audio_button_browse = tk.Button(self.root, text="Browse Audio", command=self.browse_audio)
        self.audio_button_browse.grid(row=2, column=0, pady=10, sticky="w")

        self.audio_result_label = tk.Label(self.root, text="")
        self.audio_result_label.grid(row=3, column=0, pady=10, sticky="w")

        self.audio_button_clear = tk.Button(self.root, text="Clear Audio", command=self.clear_audio_result, state="disabled")
        self.audio_button_clear.grid(row=4, column=0, pady=5, sticky="w")

        self.audio_button_classify = tk.Button(self.root, text="Classify Audio", command=self.classify_audio, state="disabled")
        self.audio_button_classify.grid(row=5, column=0, pady=5, sticky="w")

        self.image_label = tk.Label(self.root, text="Choose an image file:")
        self.image_label.grid(row=1, column=1, pady=10, sticky="w")

        self.image_button_browse = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.image_button_browse.grid(row=2, column=1, pady=10, sticky="w")

        self.image_result_label = tk.Label(self.root, text="")
        self.image_result_label.grid(row=3, column=1, pady=10, sticky="w")

        self.image_button_clear = tk.Button(self.root, text="Clear Image", command=self.clear_image_result, state="disabled")
        self.image_button_clear.grid(row=4, column=1, pady=5, sticky="w")

        self.image_button_classify = tk.Button(self.root, text="Classify Image", command=self.classify_image, state="disabled")
        self.image_button_classify.grid(row=5, column=1, pady=5, sticky="w")

    def browse_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", ".wav;.mp3;*.flac")])
        if file_path:
            self.audio_button_clear.config(state="normal")
            self.audio_button_classify.config(state="normal")
            self.audio_result_label.config(text=f"Selected File: {os.path.basename(file_path)}")
            self.current_audio_file = file_path

    def clear_audio_result(self):
        self.audio_result_label.config(text="")
        self.audio_button_clear.config(state="disabled")
        self.audio_button_classify.config(state="disabled")

    def classify_audio(self):
        try:
            prediction = self.predict_audio(self.current_audio_file)
            self.audio_result_label.config(text=f"Prediction: {'Deepfake' if prediction == 1 else 'Real'}")
        except Exception as e:
            messagebox.showerror("Error", f"Error classifying audio: {e}")

    def predict_audio(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

        X = np.array([mel_spectrogram])
        y_pred = self.audio_model.predict(X)
        predicted_class = np.argmax(y_pred)
        return predicted_class

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
        if file_path:
            self.image_button_clear.config(state="normal")
            self.image_button_classify.config(state="normal")
            self.image_result_label.config(text=f"Selected File: {os.path.basename(file_path)}")
            self.current_image_file = file_path

    def clear_image_result(self):
        self.image_result_label.config(text="")
        self.image_button_clear.config(state="disabled")
        self.image_button_classify.config(state="disabled")

    def classify_image(self):
        try:
            result = self.predict_image(self.current_image_file)
            self.image_result_label.config(text=result)
        except Exception as e:
            messagebox.showerror("Error", f"Error classifying image: {e}")

    def predict_image(self, image_path):
        img = image.load_img(image_path, target_size=(IMAGE_DIMENSIONS['height'], IMAGE_DIMENSIONS['width']))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.image_model.predict(img_array)

        result = "Real" if prediction[0, 0] < 0.5 else "Deepfake"

        return result

    def run(self):
        self.root.mainloop()

app = DeepfakeClassifier()
app.run()
