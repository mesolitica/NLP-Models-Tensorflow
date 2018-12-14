import tqdm
import os
import numpy as np
from setting import path, load_file

if not os.path.exists('spectrogram'):
    os.mkdir('spectrogram')

wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]

for fpath in tqdm.tqdm(wav_files):
    fname, spectrogram = load_file(path + fpath)
    np.save('spectrogram/{}'.format(fname.replace('wav', 'npy')), spectrogram)
