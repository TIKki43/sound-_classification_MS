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
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

data_path = '/home/timur/Documents/Projects/sound_classification/data_img'


def keras_img_prep(pathtoimg):
    img = Image.open(pathtoimg)
    img = img/255
    img = img.resize(150, 150)
    img = img.reshape(-1, 150, 150, 3)
    return img


targets = []
for img in os.listdir(data_path):
    targets.append([int(classes.get(img.split('.')[0] + '.wav'))])

targets_array = np.array(targets)
print(targets_array)

imgs = []
for img in os.listdir(data_path):
    imgs.append(load_img(data_path + '/' + img, target_size=(32, 32)))

imgs_array = np.array([img_to_array(img) for img in imgs])/255
print(imgs_array.shape)

X_train, X_val, y_train, y_val = train_test_split(imgs_array, targets_array, test_size=0.2, random_state=888)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_val = mlb.transform(y_val)

print(f"shape of X_train: {X_train.shape} \nshape of X_val: {X_val.shape}"
      f" \nshape of y_train: {y_train.shape} \nshape of y_val: {y_val.shape}")

img_rows = 32
img_cols = 32
channels = 3
classes = len(mlb.classes_)

model = Sequential()

model.add(Conv2D(64, kernel_size = (3, 3), padding = 'Same',
                     activation = 'relu',
                     input_shape = (img_rows, img_cols, channels)))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (3, 3), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


aug = ImageDataGenerator(
        rotation_range=5,
        zoom_range = 0.2,
        width_shift_range=0.2,
        height_shift_range=0.2
        )

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

batch = 32
EPOCHS = 100
history = model.fit(x=aug.flow(X_train, y_train, batch_size = batch),
                    steps_per_epoch=len(X_train)//batch,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(X_val, y_val), callbacks=[early_stop])

model.save('first.h5')
