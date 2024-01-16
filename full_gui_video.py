import tkinter as tk
from tkinter import filedialog
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tkinter import messagebox
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

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
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))
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


class ImageClassifierApp:
    def __init__(self, model_path):
        self.model = MesoInception4()
        self.model.load(model_path)

        self.root = tk.Tk()
        self.root.title("Deepfake Detection")

        
        self.root.geometry("600x300")  

        self.label = tk.Label(self.root, text="Choose an image file:")
        self.label.pack(pady=10)

        self.button_browse = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.button_browse.pack(pady=10)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_result, state="disabled")
        self.button_clear.pack(pady=5)

        self.button_classify = tk.Button(self.root, text="Classify", command=self.classify_image, state="disabled")
        self.button_classify.pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        if file_path:
            self.button_clear.config(state="normal")  
            self.button_classify.config(state="normal")  
            self.result_label.config(text=f"Selected File: {os.path.basename(file_path)}")
            global current_file
            current_file = file_path

    def clear_result(self):
        self.result_label.config(text="")
        self.button_clear.config(state="disabled")  
        self.button_classify.config(state="disabled")  

    def classify_image(self):
        try:
            result = self.predict_image(current_file)
            self.result_label.config(text=result)
        except Exception as e:
            messagebox.showerror("Error", f"Error classifying image: {e}")

    def predict_image(self, image_path):
        img = image.load_img(image_path, target_size=(image_dimensions['height'], image_dimensions['width']))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.model.predict(img_array)

        result = "Real" if prediction[0, 0] < 0.5 else "Deepfake"

        return result

    def run(self):
        self.root.mainloop()


app = ImageClassifierApp(model_path='D:\\Desktop\\hack\\MesoInception_F2F.h5')
app.run()