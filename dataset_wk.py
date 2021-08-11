import gc
import os
import math
import pylab
import librosa
import numpy as np
import pandas as pd
import librosa.display
import keras.backend as K
import matplotlib.pyplot as plt

from path import Path
from glob import glob
from keras import models
from matplotlib import figure
from pydub import AudioSegment
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU

# fbeta
path_to_data = '/home/timur/Documents/Projects/sound_classification/dataset'


def create_spectrogram(filename, path):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)

    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, clip, sample_rate, fig, ax, S


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export('/home/timur/Documents/Projects/sound_classification/dataset/audio/' + split_filename,
                           format="wav")

    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 1)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully ' + self.filename)


files = os.listdir("/home/timur/Documents/Projects/sound_classification/dataset/audio")

folder = '/home/timur/Documents/Projects/sound_classification/dataset/audio'

for file in files:
    split_wav = SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=1)

Data_dir=np.array(glob(r"D:\reposetory\Save_Transport\dataset\test\one_sec_noise\*"))

i=0
for file in Data_dir[i:i+2000]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    path = r'/home/timur/Documents/Projects/sound_classification/img.jpg' + name + '.jpg'
    create_spectrogram(filename, path)
print(gc.collect())

path_to_data = '/home/timur/Documents/Projects/sound_classification/dataset/audio/audio/16000/'

for num_sound in range(len(os.listdir(path_to_data))):
    create_spectrogram(path_to_data + os.listdir(path_to_data)[num_sound],
                       f'data_img/{(os.listdir(path_to_data)[num_sound]).split(".")[0]}.png')

