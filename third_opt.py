import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

df = pd.read_csv('/home/timur/Documents/Projects/sound_classification/ag_files/esc50.csv')
df_10 = df[df['esc10'] == True]
df_10 = df_10.drop(['fold', 'esc10', 'src_file', 'take'], axis=1)
print(df_10)
classes = df_10['category'].unique()
df_10['target'] = df_10['category'].map({i:x for x,i in enumerate(classes)})
sample_df = df_10.drop_duplicates(subset=['target'])
signals = {}
mel_spectrograms = {}
mfccs = {}

for row in tqdm(sample_df.iterrows()):  # every row will be like [[index], [filename , target , category]]
    signal, rate = librosa.load('/home/timur/Documents/Projects/sound_classification/dataset/audio/audio/44100/' + row[1][0])
    signals[row[1][2]] = signal  # row[1][2] will be the category of that signal. eg. signal["dog"] = signal of dog sound

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=2048, hop_length=512)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # visualizing mel_spectrogram directly gives black image. So, coverting from power_to_db is required
    mel_spectrograms[row[1][2]] = mel_spec

    mfcc = librosa.feature.mfcc(signal, rate, n_mfcc=13, dct_type=3)
    mfccs[row[1][2]] = mfcc


X, y = [], []
for data in tqdm(df_10.iterrows()):
    sig, sr = librosa.load('/home/timur/Documents/Projects/sound_classification/dataset/audio/audio/44100/' + data[1][0])
    for i in range(3):
        n = np.random.randint(0, len(sig) - (sr * 2))
        sig_ = sig[n:int(n + (sr * 2))]
        mfcc_ = librosa.feature.mfcc(sig_, sr=sr, n_mfcc=13)
        X.append(mfcc_)
        y.append(data[1][1])

# convert list to numpy array
X = np.array(X)
y = np.array(y)

#one-hot encoding the target
y = tf.keras.utils.to_categorical(y, num_classes=10)

# our tensorflow model takes input as (no_of_sample , height , width , channel).
# here X has dimension (no_of_sample , height , width).
# So, the below code will reshape it to (no_of_sample , height , width , 1).
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

x_train , x_val , y_train , y_val = train_test_split(X , y ,test_size=0.2, random_state=2020)

INPUTSHAPE = (13, 87, 1)

model = models.Sequential([
                          layers.Conv2D(16, (3, 3),activation = 'relu', padding='valid', input_shape=INPUTSHAPE),
                          layers.Conv2D(16, (3, 3), activation='relu', padding='valid'),

                          layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
                          layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),

                          layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
                          layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
                          layers.GlobalAveragePooling2D(),


                          layers.Dense(32, activation='relu'),
                          layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

history = model.fit(x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            callbacks=[early_stop])