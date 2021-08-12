import os

import cv2
import sys
sys.path.append('../')
import numpy as np
import pandas as pd

sys.path.insert(1, '/home/timur/Documents/Projects/sound_classification/ag_files')
from ag_files.data_prep import classes
from PIL import Image, ImageFont, ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

data_path = '/home/timur/Documents/Projects/sound_classification/data_img'
# print(classes.get('1-137-A-32.wav'))


def img_prep(pathtoimg):
    img = Image.open(pathtoimg)
    img = img/255
    img = img.resize(150, 150)
    img = img.reshape(-1, 150, 150, 3)
    return img


targets = []
for img in os.listdir(data_path):
    targets.append(int(classes.get(img.split('.')[0] + '.wav')))

targets_array = np.array(targets)
print(targets_array)

imgs = []
for img in os.listdir(data_path):
    imgs.append(load_img(data_path + '/' + img, target_size=(32, 32)))

imgs_array = np.array([img_to_array(img) for img in imgs])/255
print(imgs_array.shape)


X_train, X_val, y_train, y_val = train_test_split(imgs_array, targets_array,
                                                test_size=0.2,
                                                random_state=888)


