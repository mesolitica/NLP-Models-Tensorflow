import numpy as np
import librosa
import copy
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
import os
import unicodedata
import re

# P: Padding
# S: Start of Sentence
# E: End of Sentence
path = '../data/'
vocab = "PSE abcdefghijklmnopqrstuvwxyz'.?"
max_duration = 10.0
sample_rate = 22050
fourier_window_size = 2048
frame_shift = 0.0125
frame_length = 0.05
hop_length = int(sample_rate * frame_shift)
win_length = int(sample_rate * frame_length)
n_mels = 80
power = 1.2
iteration_griffin = 50
preemphasis = 0.97
max_db = 100
ref_db = 20
embed_size = 256
encoder_num_banks = 16
decoder_num_banks = 8
num_highwaynet_blocks = 4
resampled = 5
dropout_rate = 0.5
learning_rate = 0.001
batch_size = 32


def get_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr = sample_rate)
    y, _ = librosa.effects.trim(y)
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])
    linear = librosa.stft(
        y = y,
        n_fft = fourier_window_size,
        hop_length = hop_length,
        win_length = win_length,
    )
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(sample_rate, fourier_window_size, n_mels)
    mel = np.dot(mel_basis, mag)
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)
    return mel.T.astype(np.float32), mag.T.astype(np.float32)


def invert_spectrogram(spectrogram):
    return librosa.istft(
        spectrogram, hop_length, win_length = win_length, window = 'hann'
    )


def spectrogram2wav(mag):
    mag = mag.T
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db
    mag = np.power(10.0, mag * 0.05)
    wav = griffin_lim(mag)
    wav = signal.lfilter([1], [1, -preemphasis], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(iteration_griffin):
        X_T = invert_spectrogram(X_best)
        est = librosa.stft(
            X_T, fourier_window_size, hop_length, win_length = win_length
        )
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_T = invert_spectrogram(X_best)
    return np.real(X_T)


def get_wav(spectrogram):
    mag = (np.clip(spectrogram.T, 0, 1) * max_db) - max_db + ref_db
    mag = np.power(10.0, mag * 0.05)
    wav = griffin_lim(mag)
    wav = signal.lfilter([1], [1, -preemphasis], wav)
    return librosa.effects.trim(wav).astype(np.float32)


def load_file(path):
    fname = os.path.basename(path)
    mel, mag = get_spectrogram(path)
    t = mel.shape[0]
    num_paddings = resampled - (t % resampled) if t % resampled != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode = 'constant')
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode = 'constant')
    return fname, mel.reshape((-1, n_mels * resampled)), mag


def text_normalize(text):
    text = ''.join(
        char
        for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    text = text.lower()
    text = re.sub('[^{}]'.format(vocab), ' ', text)
    text = re.sub('[ ]+', ' ', text)
    return text


def get_cached(path):
    mel = 'mel/{}.npy'.format(path)
    mag = 'mag/{}.npy'.format(path)
    return np.load(mel), np.load(mag)

def plot_alignment(alignment, e):
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)
    fig.colorbar(im)
    plt.title('epoch %d' % (e))
    plt.show()

char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}
