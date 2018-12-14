# coding: utf-8

# In[1]:


import tensorflow as tf
from setting import (
    text2idx,
    get_cached,
    batch_size,
    n_mels,
    reduction_factor,
    idx2char,
)
from tqdm import tqdm
import numpy as np
import os


# In[2]:


paths, lengths, texts = [], [], []
text_files = [f for f in os.listdir('spectrogram') if f.endswith('.npy')]
for fpath in text_files:
    with open('../data/' + fpath.replace('npy', 'txt')) as fopen:
        text, converted = text2idx(fopen.read())
    texts.append(converted)
    lengths.append(len(text))
    paths.append(fpath.replace('.npy', ''))


# In[3]:


def dynamic_batching(paths):
    spectrograms, max_x = [], 0
    for path in paths:
        spectrograms.append(np.load('spectrogram/' + path + '.npy'))
        if spectrograms[-1].shape[0] > max_x:
            max_x = spectrograms[-1].shape[0]
    return spectrograms, max_x


# In[4]:


from model import Model

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[5]:


for e in range(30):
    pbar = tqdm(range(0, len(text_files), batch_size), desc = 'minibatch loop')
    total_cost, total_acc = 0, 0
    for k in pbar:
        index = min(k + batch_size, len(text_files))
        files, max_x = dynamic_batching(paths[k:index])
        max_y = max(lengths[k:index])
        batch_x = np.zeros((len(files), max_x, n_mels * reduction_factor))
        batch_y = np.zeros((len(files), max_y))
        for n in range(len(files)):
            batch_x[n] = np.pad(
                files[n],
                ((max_x - files[n].shape[0], 0), (0, 0)),
                mode = 'constant',
            )
            batch_y[n] = np.pad(
                texts[k + n],
                ((0, max_y - len(texts[k + n]))),
                mode = 'constant',
            )
        _, acc, cost = sess.run(
            [model.optimizer, model.accuracy, model.cost],
            feed_dict = {
                model.X: batch_x,
                model.Y: batch_y,
                model.Y_seq_len: lengths[k:index],
            },
        )
        total_cost += cost
        total_acc += acc
        pbar.set_postfix(cost = cost, accuracy = acc)
    total_cost /= len(text_files) / batch_size
    total_acc /= len(text_files) / batch_size

    print('epoch %d, avg loss %f, avg acc %f' % (e + 1, total_cost, total_acc))


empty_y = np.zeros((1, len(batch_y[0])))
predicted = ''.join(
    [
        idx2char[c]
        for c in sess.run(
            model.preds, feed_dict = {model.X: batch_x[:1], model.Y: empty_y}
        )[0]
        if idx2char[c] not in ['S', 'E']
    ]
)
ground_truth = ''.join(
    [idx2char[c] for c in batch_y[0] if idx2char[c] not in ['S', 'E']]
)
print('predicted: %s, ground truth: %s' % (predicted, ground_truth))
