from utils import ESC50
import pandas as pd
import numpy as np

train_splits = [1,2,3,4]
test_split = 5

shared_params = {'csv_path': '/home/timur/Documents/Projects/sound_classification/ag_files/esc50.csv',
                 'wav_dir': '/home/timur/Documents/Projects/sound_classification/dataset/audio/audio',
                 'dest_dir': '/home/timur/Documents/Projects/sound_classification/dataset/audio/audio/16000',
                 'audio_rate': 16000,
                 'only_ESC10': True,
                 'pad': 0,
                 'normalize': True}

train_gen = ESC50(folds=train_splits,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=True,
                  **shared_params).batch_gen(16)

test_gen = ESC50(folds=[test_split],
                 randomize=False,
                 strongAugment=False,
                 random_crop=False,
                 inputLength=4,
                 mix=False,
                 **shared_params).batch_gen(16)

X, Y = next(train_gen)
print(X.shape, Y.shape)



df = pd.read_csv('/home/timur/Documents/Projects/sound_classification/ag_files/esc50.csv')
classes = df[['filename', 'target']].values.tolist()
classes = set(['{} {}'.format(c[0], c[1]) for c in classes])
classes = np.array([c.split(' ') for c in classes])
classes = {k: v for k, v in classes}
print(classes)

