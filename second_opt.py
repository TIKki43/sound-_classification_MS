import splitfolders
import sys
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
sys.path.append('../')
import numpy as np
import pandas as pd
import shutil

sys.path.insert(1, '/home/timur/Documents/Projects/sound_classification/ag_files')
from ag_files.data_prep import classes
import os, glob, numpy as np
from PIL import Image

dirname = '/home/timur/Documents/Projects/sound_classification/im_data'
data_path = '/home/timur/Documents/Projects/sound_classification/data_img'


def create_dir(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)


for i in classes.values():
    create_dir(dirname + '/' + i)

for img in os.listdir(data_path):
    for clas in os.listdir(dirname):
        try:
            shutil.move(data_path + '/' + img, dirname + '/' + classes.get(img[:-3] + 'wav') + '/' + img)
        except:
            pass
print('OK')

splitfolders.ratio('/home/timur/Documents/Projects/sound_classification/im_data',
                   output="/home/timur/Documents/Projects/sound_classification/spl_dt", seed=1337, ratio=(.8, 0.1,0.1))

batch = 32
channels = 3
row, col = 32, 32

train_data = ImageDataGenerator(rescale=1/255, shear_range=0.4, zoom_range=0.4, horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1/255)

train_gen = train_data.flow_from_directory('/home/timur/Documents/Projects/sound_classification/spl_dt/train',
                                           target_size=(row, col), batch_size=batch,
                                           color_mode='rgb', class_mode='categorical')

test_gen = test_data.flow_from_directory('/home/timur/Documents/Projects/sound_classification/spl_dt/val',
                                         target_size=(row, col), batch_size=batch,
                                         color_mode='rgb', class_mode='categorical')

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding='Same',
                     activation='relu',
                     input_shape=(row, col, channels)))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='softmax'))


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_gen, epochs=100, validation_data=test_gen, callbacks=[early_stop])

model.save('second.h5')