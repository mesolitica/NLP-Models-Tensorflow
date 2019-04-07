import tqdm
import os
import numpy as np
from utils import load_file, path

if not os.path.exists('mel'):
    os.mkdir('mel')
if not os.path.exists('mag'):
    os.mkdir('mag')
wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
for fpath in tqdm.tqdm(wav_files):
    fname, mel, mag = load_file(path + fpath)
    np.save('mel/{}'.format(fname.replace('wav', 'npy')), mel)
    np.save('mag/{}'.format(fname.replace('wav', 'npy')), mag)
