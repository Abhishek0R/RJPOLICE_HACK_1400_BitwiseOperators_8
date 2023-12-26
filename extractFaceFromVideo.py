import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
plt.style.use('ggplot')

train_dir = '/content/drive/My Drive/Meso/Test/test_videos/'
train_video_files = [train_dir + i for i in os.listdir(train_dir)]
print(train_video_files)

from mtcnn.mtcnn import MTCNN
detector = MTCNN()
v_cap = cv2.VideoCapture(train_video_files[0])
_, frame = v_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(frame, cmap = 'gray')
plt.axis('off')
result = detector.detect_faces(frame)
print(result)
print(len(result))

def img_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hist_equalized_image = clahe.apply(image)
    return(hist_equalized_image)
for j in range(len(result)):
    bounding_box = result[j]['box']
    cv2.rectangle(frame,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)
    plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.axis("off")
    plt.show()
    frame_cropped = frame[bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0] : bounding_box[0] + bounding_box[2]]
    plt.figure(figsize=(12, 8))
    im = cv2.cvtColor(frame_cropped,cv2.COLOR_RGB2GRAY)
    im = img_enhancement(im)
    plt.imshow(im,cmap = 'gray')
    plt.axis("off")
    plt.show()
    