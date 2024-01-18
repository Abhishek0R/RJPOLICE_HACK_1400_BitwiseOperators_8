
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import imageio

# Load ResNet model
model_path = "D:\\Desktop\\hack\\resnet.h5"
model = tf.keras.models.load_model(model_path)

# Define classes
CLASSES = ['Real', 'Fake']

class ResNetClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ResNet Image Classifier")

        self.setup_gui()

    def setup_gui(self):
        self.image_label = tk.Label(self.root, text="No Image")
        self.image_label.pack()

        self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack()

        self.classify_button = tk.Button(self.root, text="Classify", command=self.classify_image, state="disabled")
        self.classify_button.pack()

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
        if file_path:
            self.display_image(file_path)
            self.classify_button.config(state="normal")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

        self.image_path = file_path

    def classify_image(self):
        if hasattr(self, 'image_path'):
            image = imageio.imread(self.image_path)

            img = Image.fromarray(image).resize((224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(img)
            input_arr = np.array([input_arr])

            predict = model.predict(input_arr)
            probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
            predict = probability_model.predict(input_arr)

            predicted_class = CLASSES[np.argmax(predict[0])]

            self.result_label.config(text=f"Prediction: {predicted_class}")
        else:
            self.result_label.config(text="Please browse an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResNetClassifierApp(root)
    root.mainloop()
