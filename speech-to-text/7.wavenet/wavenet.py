# coding: utf-8

# In[1]:


import librosa
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm


# In[2]:


wav_files = [f for f in os.listdir('./data') if f.endswith('.wav')]
text_files = [f for f in os.listdir('./data') if f.endswith('.txt')]


# In[3]:


inputs, targets = [], []
for (wav_file, text_file) in tqdm(
    zip(wav_files, text_files), total = len(wav_files), ncols = 80
):
    path = './data/' + wav_file
    try:
        y, sr = librosa.load(path, sr = None)
    except:
        continue
    inputs.append(
        librosa.feature.mfcc(
            y = y, sr = sr, n_mfcc = 40, hop_length = int(1e-1 * sr)
        ).T
    )
    with open('./data/' + text_file) as f:
        targets.append(f.read())


# In[4]:


inputs = tf.keras.preprocessing.sequence.pad_sequences(
    inputs, dtype = 'float32', padding = 'post'
)

chars = list(set([c for target in targets for c in target]))
num_classes = len(chars) + 1

idx2char = {idx: char for idx, char in enumerate(chars)}
char2idx = {char: idx for idx, char in idx2char.items()}

targets = [[char2idx[c] for c in target] for target in targets]


# In[5]:


def sparse_tuple_from(sequences, dtype = np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype = np.int64)
    values = np.asarray(values, dtype = dtype)
    shape = np.asarray(
        [len(sequences), np.asarray(indices).max(0)[1] + 1], dtype = np.int64
    )

    return indices, values, shape


# In[10]:


def pad_causal(x, size, rate):
    pad_len = (size - 1) * rate
    return tf.pad(x, [[0, 0], [pad_len, 0], [0, 0]])


class Model:
    def __init__(
        self,
        num_layers,
        size_layers,
        learning_rate,
        num_features,
        num_blocks = 3,
        block_size = 128,
        dropout = 1.0,
    ):
        self.X = tf.placeholder(
            tf.float32, [None, inputs.shape[1], num_features]
        )
        self.Y = tf.sparse_placeholder(tf.int32)
        seq_lens = tf.fill([tf.shape(self.X)[0]], inputs.shape[1])

        def residual_block(x, size, rate, block):
            with tf.variable_scope(
                'block_%d_%d' % (block, rate), reuse = False
            ):
                conv_filter = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size = size,
                    strides = 1,
                    padding = 'same',
                    dilation_rate = rate,
                    activation = tf.nn.tanh,
                )
                conv_gate = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size = size,
                    strides = 1,
                    padding = 'same',
                    dilation_rate = rate,
                    activation = tf.nn.sigmoid,
                )
                out = tf.multiply(conv_filter, conv_gate)
                out = tf.layers.conv1d(
                    out,
                    block_size,
                    kernel_size = 1,
                    strides = 1,
                    padding = 'same',
                    activation = tf.nn.tanh,
                )
                return tf.add(x, out), out

        forward = tf.layers.conv1d(
            self.X, block_size, kernel_size = 1, strides = 1, padding = 'SAME'
        )
        zeros = tf.zeros_like(forward)
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                forward, s = residual_block(
                    forward, size = 7, rate = r, block = i
                )
                zeros = tf.add(zeros, s)
        forward = tf.layers.conv1d(
            zeros,
            block_size,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME',
            activation = tf.nn.tanh,
        )
        logits = tf.layers.conv1d(
            forward, num_classes, kernel_size = 1, strides = 1, padding = 'SAME'
        )
        time_major = tf.transpose(logits, [1, 0, 2])
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(time_major, seq_lens)
        decoded = tf.to_int32(decoded[0])
        self.preds = tf.sparse.to_dense(decoded)
        self.cost = tf.reduce_mean(
            tf.nn.ctc_loss(
                self.Y,
                time_major,
                seq_lens,
                ignore_longer_outputs_than_inputs = True,
            )
        )
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(self.cost)


# In[11]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

size_layers = 128
learning_rate = 1e-3
num_layers = 2

model = Model(num_layers, size_layers, learning_rate, inputs.shape[2])
sess.run(tf.global_variables_initializer())


# In[12]:


import time

batch_size = 32

for e in range(50):
    lasttime = time.time()
    pbar = tqdm(
        range(0, len(inputs), batch_size), desc = 'minibatch loop', ncols = 80
    )
    for i in pbar:
        batch_x = inputs[i : min(i + batch_size, len(inputs))]
        batch_y = sparse_tuple_from(
            targets[i : min(i + batch_size, len(inputs))]
        )
        _, cost = sess.run(
            [model.optimizer, model.cost],
            feed_dict = {model.X: batch_x, model.Y: batch_y},
        )
        pbar.set_postfix(cost = cost)


# In[ ]:


import random

random_index = random.randint(0, len(targets) - 1)
batch_x = inputs[random_index : random_index + 1]
print(
    'real:',
    ''.join(
        [idx2char[no] for no in targets[random_index : random_index + 1][0]]
    ),
)
batch_y = sparse_tuple_from(targets[random_index : random_index + 1])
pred = sess.run(model.preds, feed_dict = {model.X: batch_x})[0]
print('predicted:', ''.join([idx2char[no] for no in pred]))
