#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import time
import random
import os


# In[ ]:


trainset = sklearn.datasets.load_files(container_path = 'data', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset,1.0)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))


# In[ ]:


concat = ' '.join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print('vocab from size: %d'%(vocabulary_size))
print('Most common words', count[4:10])
print('Sample data', data[:10], [rev_dictionary[i] for i in data[:10]])


# In[ ]:


GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


# In[ ]:


embedding_size = 128
dimension_output = len(trainset.target_names)
maxlen = 50
batch_size = 32
kernel_size = 3
num_filters = 150


# In[ ]:


class Model:
    def __init__(self, 
                 maxlen,
                 dimension_output,
                 vocab_size,
                 embedding_size,
                 kernel_size,
                 num_filters,
                 learning_rate):
        self.X = tf.placeholder(tf.int32,[None, maxlen])
        self.Y = tf.placeholder(tf.int32,[None])
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
        embedded = tf.nn.embedding_lookup(embeddings, self.X)
        first_region = tf.layers.conv1d(
                    embedded,
                    num_filters,
                    kernel_size = kernel_size,
                    strides = 1,
                    padding = 'valid'
                )
        forward = tf.nn.relu(first_region)
        forward = tf.layers.conv1d(
                    forward,
                    num_filters,
                    kernel_size = kernel_size,
                    strides = 1,
                    padding = 'same'
                )
        forward = tf.layers.batch_normalization(forward)
        forward = tf.nn.relu(first_region)
        forward = tf.layers.conv1d(
                    forward,
                    num_filters,
                    kernel_size = kernel_size,
                    strides = 1,
                    padding = 'same'
                )
        forward = tf.layers.batch_normalization(forward)
        forward = tf.nn.relu(first_region)
        forward = forward + first_region
        
        def _block(x):
            x = tf.pad(x, paddings=[[0, 0], [0, 1], [0, 0]])
            px = tf.layers.max_pooling1d(x, 3, 2)
            x = tf.nn.relu(px)
            x = tf.layers.conv1d(
                    x,
                    num_filters,
                    kernel_size = kernel_size,
                    strides = 1,
                    padding = 'same'
                )
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(
                    x,
                    num_filters,
                    kernel_size = kernel_size,
                    strides = 1,
                    padding = 'same'
                )
            x = tf.layers.batch_normalization(x)
            x = x + px
            return x
        while forward.get_shape().as_list()[1] >= 2:
            forward = _block(forward)
        self.logits = tf.reduce_sum(tf.layers.conv1d(
            forward, dimension_output, kernel_size = 1, strides = 1, padding = 'SAME'
        ), 1)
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1,output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(maxlen, dimension_output, len(dictionary), embedding_size,
             kernel_size, num_filters, 1e-3)
sess.run(tf.global_variables_initializer())


# In[ ]:


vectors = str_idx(trainset.data,dictionary,maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target,test_size = 0.2)


# In[ ]:


from tqdm import tqdm
import time

EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n' % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(
        range(0, len(train_X), batch_size), desc = 'train minibatch loop'
    )
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        batch_x_expand = np.expand_dims(batch_x,axis = 1)
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict = {
                model.Y: batch_y,
                model.X: batch_x
            },
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost = cost, accuracy = acc)
        
    pbar = tqdm(range(0, len(test_X), batch_size), desc = 'test minibatch loop')
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        batch_x_expand = np.expand_dims(batch_x,axis = 1)
        acc, cost = sess.run(
            [model.accuracy, model.cost],
            feed_dict = {
                model.Y: batch_y,
                model.X: batch_x
            },
        )
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost = cost, accuracy = acc)

    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size

    if test_acc > CURRENT_ACC:
        print(
            'epoch: %d, pass acc: %f, current acc: %f'
            % (EPOCH, CURRENT_ACC, test_acc)
        )
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print('time taken:', time.time() - lasttime)
    print(
        'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )
    EPOCH += 1


# In[ ]:


real_Y, predict_Y = [], []

pbar = tqdm(
    range(0, len(test_X), batch_size), desc = 'validation minibatch loop'
)
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    predict_Y += np.argmax(
        sess.run(
            model.logits, feed_dict = {model.X: batch_x, model.Y: batch_y}
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[ ]:


print(metrics.classification_report(real_Y, predict_Y, target_names = trainset.target_names))

