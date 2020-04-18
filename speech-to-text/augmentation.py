import numpy as np
import librosa
import os
import scipy
import json


def change_pitch_speech(samples):
    y_pitch_speed = samples.copy()
    length_change = np.random.uniform(low = 0.8, high = 1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(
        np.arange(0, len(y_pitch_speed), speed_fac),
        np.arange(0, len(y_pitch_speed)),
        y_pitch_speed,
    )
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_amplitude(samples):
    y_aug = samples.copy()
    dyn_change = np.random.uniform(low = 1.5, high = 3)
    return y_aug * dyn_change


def add_noise(samples):
    y_noise = samples.copy()
    noise_amp = 0.01 * np.random.uniform() * np.amax(y_noise)
    return y_noise.astype('float64') + noise_amp * np.random.normal(
        size = y_noise.shape[0]
    )


def add_hpss(samples):
    y_hpss = librosa.effects.hpss(samples.astype('float64'))
    return y_hpss[1]


def strech(samples):
    input_length = len(samples)
    streching = samples.copy()
    random_strech = np.random.uniform(low = 0.5, high = 1.3)
    print('random_strech = ', random_strech)
    streching = librosa.effects.time_stretch(
        streching.astype('float'), random_strech
    )
    return streching


def random_augmentation(samples):
    cp = samples.copy()
    if np.random.randint(0, 2):
        length_change = np.random.uniform(low = 0.8, high = 1)
        speed_fac = 1.0 / length_change
        print('resample length_change = ', length_change)
        tmp = np.interp(
            np.arange(0, len(cp), speed_fac), np.arange(0, len(cp)), cp
        )
        minlen = min(cp.shape[0], tmp.shape[0])
        cp *= 0
        cp[0:minlen] = tmp[0:minlen]

    if np.random.randint(0, 2):
        dyn_change = np.random.uniform(low = 1.5, high = 3)
        print('dyn_change = ', dyn_change)
        cp = cp * dyn_change

    if np.random.randint(0, 2):
        noise_amp = 0.005 * np.random.uniform() * np.amax(cp)
        cp = cp.astype('float64') + noise_amp * np.random.normal(
            size = cp.shape[0]
        )

    if np.random.randint(0, 2):
        timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)
        print('timeshift_fac = ', timeshift_fac)
        start = int(cp.shape[0] * timeshift_fac)
        if start > 0:
            cp = np.pad(cp, (start, 0), mode = 'constant')[0 : cp.shape[0]]
        else:
            cp = np.pad(cp, (0, -start), mode = 'constant')[0 : cp.shape[0]]
    return cp


with open('train-test.json') as fopen:
    wavs = json.load(fopen)['train']
    
if not os.path.exists('augment'):
    os.makedirs('augment')

for no, wav in enumerate(wavs):
    try:
        root, ext = os.path.splitext(wav)
        if (no + 1) % 100 == 0:
            print(no + 1, root, ext)
        root = root.replace('/', '<>')
        root = '%s/%s'%('augment', root)
        sample_rate, samples = scipy.io.wavfile.read(wav)
        aug = change_pitch_speech(samples)
        librosa.output.write_wav(
            '%s-1%s' % (root, ext),
            aug.astype('float32'),
            sample_rate,
            norm = True,
        )

        aug = change_amplitude(samples)
        librosa.output.write_wav(
            '%s-2%s' % (root, ext),
            aug.astype('float32'),
            sample_rate,
            norm = True,
        )

        aug = add_noise(samples)
        librosa.output.write_wav(
            '%s-3%s' % (root, ext),
            aug.astype('float32'),
            sample_rate,
            norm = True,
        )

        aug = add_hpss(samples)
        librosa.output.write_wav(
            '%s-4%s' % (root, ext),
            aug.astype('float32'),
            sample_rate,
            norm = True,
        )

        aug = strech(samples)
        librosa.output.write_wav(
            '%s-5%s' % (root, ext),
            aug.astype('float32'),
            sample_rate,
            norm = True,
        )

        aug = random_augmentation(samples)
        librosa.output.write_wav(
            '%s-6%s' % (root, ext),
            aug.astype('float32'),
            sample_rate,
            norm = True,
        )
    except Exception as e:
        print(e)
        pass