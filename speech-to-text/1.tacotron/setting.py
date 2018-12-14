path = '../data/'
max_len = 100
sampling_rate = 22050
n_fft = 2048
frame_shift = 0.0125
frame_length = 0.05
hop_length = int(sampling_rate * frame_shift)
win_length = int(sampling_rate * frame_length)
n_mels = 80

embed_size = 256
encoder_num_banks = 16
decoder_num_banks = 8
num_highway_blocks = 4
reduction_factor = 5

learning_rate = 1e-4
batch_size = 32

vocab = "ES abcdefghijklmnopqrstuvwxyz'"
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

import re
import os


def text2idx(text):
    text = re.sub(r'[^a-z ]', '', text.lower()).strip() + 'S'
    converted = [char2idx[char] for char in text]
    return text, converted


import librosa
import numpy as np


def get_spectrogram(fpath):
    y, sr = librosa.load(fpath, sr = sampling_rate)
    D = librosa.stft(
        y = y, n_fft = n_fft, hop_length = hop_length, win_length = win_length
    )
    magnitude = np.abs(D)
    power = magnitude ** 2
    S = librosa.feature.melspectrogram(S = power, n_mels = n_mels)
    return np.transpose(S.astype(np.float32))


def reduce_frames(x, r_factor):
    T, C = x.shape
    num_paddings = reduction_factor - (T % r_factor) if T % r_factor != 0 else 0
    padded = np.pad(x, [[0, num_paddings], [0, 0]], 'constant')
    return np.reshape(padded, (-1, C * r_factor))


def restore_shape(x, r_factor):
    N, _, C = x.shape
    return x.reshape((N, -1, C // r_factor))


def load_file(path):
    fname = os.path.basename(path)
    spectrogram = get_spectrogram(path)
    spectrogram = reduce_frames(spectrogram, reduction_factor)
    return fname, spectrogram


def get_cached(path):
    spectrogram = 'spectrogram/' + path + '.npy'
    return np.load(spectrogram)
