#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os


# In[2]:


def build_dataset(words, n_words, atleast=1):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# In[3]:


lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
    
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return ' '.join([i.strip() for i in filter(None, text.split())])

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))
    
min_line_length = 2
max_line_length = 5
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

question_test = short_questions[500:550]
answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]


# In[4]:


concat_from = ' '.join(short_questions+question_test).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from)
print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10])
print('Sample data', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print('filtered vocab size:',len(dictionary_from))
print("% of vocab used: {}%".format(round(len(dictionary_from)/vocabulary_size_from,4)*100))


# In[5]:


concat_to = ' '.join(short_answers+answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print('vocab from size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print('filtered vocab size:',len(dictionary_to))
print("% of vocab used: {}%".format(round(len(dictionary_to)/vocabulary_size_to,4)*100))


# In[6]:


GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']


# In[7]:


for i in range(len(short_answers)):
    short_answers[i] += ' EOS'


# In[41]:


def squash(X, epsilon = 1e-9):
    vec_squared_norm = tf.reduce_sum(tf.square(X), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    return scalar_factor * X

def conv_layer(X, num_output, num_vector, kernel=None, stride=None):
    capsules = tf.layers.conv1d(X, num_output * num_vector,
                                kernel, stride, padding="VALID", activation=tf.nn.relu)
    capsules = tf.reshape(capsules, (tf.shape(X)[0], -1, num_vector, 1))
    return squash(capsules)

def routing(X, b_IJ, seq_len, dimension_out, routing_times = 2):
    shape_X = tf.shape(X)[1]
    w = tf.Variable(tf.truncated_normal([1, 1, seq_len, 4, dimension_out//2], stddev=1e-1))
    X = tf.tile(X, [1, 1, seq_len, 1, dimension_out])
    w = tf.tile(w, [tf.shape(X)[0], tf.shape(X)[1], 1, 1, routing_times])
    print('X shape: %s, w shape: %s'%(str(X.shape), str(w.shape)))
    u_hat = tf.matmul(w, X, transpose_a=True)
    u_hat_stopped = tf.stop_gradient(u_hat)
    for i in range(routing_times):
        c_IJ = tf.nn.softmax(b_IJ, dim=2)
        print(c_IJ)
        if i == routing_times - 1:
            s_J = tf.multiply(c_IJ, u_hat)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            v_J = squash(s_J)
        else:
            s_J = tf.multiply(c_IJ, u_hat_stopped)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            v_J = squash(s_J)
            v_J_tiled = tf.tile(v_J, [1, shape_X, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
            b_IJ += u_produce_v
    return v_J

def fully_conn_layer(X, num_output, dimension_out):
    batch_size = tf.shape(X)[1]
    X_ = tf.reshape(X, shape=(tf.shape(X)[0], -1, 1, X.shape[-2].value, 1))
    b_IJ = tf.fill([tf.shape(X)[0], tf.shape(X)[1], num_output, 1, 1], 0.0)
    capsules = routing(X_, b_IJ, num_output, dimension_out, routing_times = 2)
    capsules = tf.squeeze(capsules, axis=1)
    return capsules

class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen,
                 from_dict_size, to_dict_size, learning_rate, batch_size,
                kernels=[2, 4, 4],strides=[3,2,1],epsilon=1e-8,
                force_teaching_ratio=0.5,beam_width = 5):
        
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer,initializer=tf.orthogonal_initializer(),reuse=reuse)
        
        self.X = tf.placeholder(tf.int32, [None, maxlen])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        
        encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, self.X)
        
        results = []
        for i in range(len(kernels)):
            conv = tf.layers.conv1d(encoder_embedded, filters=32,
                                    kernel_size=kernels[i], strides=strides[i],
                                    padding='VALID')
            caps1 = conv_layer(conv, 4, 4, kernels[i], strides[i])
            caps2 = fully_conn_layer(caps1,seq_len,embedded_size)
            v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2),axis=2, keep_dims=True) + epsilon)[:,:,0,:]
            print('output shape: %s\n'%(str(v_length.shape)))
            results.append(v_length)
        results = tf.concat(results,1)
        self.X_seq_len = tf.fill([batch_size], seq_len * len(kernels))
        
        self.encoder_out, encoder_state = tf.nn.dynamic_rnn(
            cell = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)]), 
            inputs = results,
            dtype = tf.float32)
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        dense = tf.layers.Dense(to_dict_size)
        
        with tf.variable_scope('decode'):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = size_layer, 
                                                                memory = self.encoder_out,
                                                                memory_sequence_length = self.X_seq_len)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)]),
                attention_mechanism = attention_mechanism,
                attention_layer_size = size_layer)
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, decoder_input),
                sequence_length = self.Y_seq_len,
                embedding = decoder_embedding,
                sampling_probability = 1 - force_teaching_ratio,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = training_helper,
                initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
                output_layer = dense)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
            X_seq_len_tiled = tf.contrib.seq2seq.tile_batch(self.X_seq_len, beam_width)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = size_layer, 
                memory = encoder_out_tiled,
                memory_sequence_length = X_seq_len_tiled)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([cells(reuse=True) for _ in range(num_layers)]),
                attention_mechanism = attention_mechanism,
                attention_layer_size = size_layer)
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = decoder_cell,
                embedding = decoder_embedding,
                start_tokens = tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
                end_token = EOS,
                initial_state = decoder_cell.zero_state(batch_size * beam_width, tf.float32).clone(cell_state = encoder_state_tiled),
                beam_width = beam_width,
                output_layer = dense,
                length_penalty_weight = 0.0)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = False,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]
            
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[17]:


size_layer = 128
num_layers = 2
embedded_size = 128
learning_rate = 1e-2
batch_size = 16
epoch = 20
maxlen = 10


# In[42]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, 5, maxlen, len(dictionary_from), 
                len(dictionary_to), learning_rate, batch_size)
sess.run(tf.global_variables_initializer())


# In[ ]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
        X.append(ints)
    return X


# In[ ]:


X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)


# In[ ]:


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens

def pad_sentence_batch_static(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


# In[ ]:


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    X, Y = shuffle(X, Y)
    for k in range(0, len(short_questions), batch_size):
        index = min(k + batch_size, len(short_questions))
        batch_x, _ = pad_sentence_batch_static(X[k: index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD)
        predicted, loss, _, accuracy = sess.run([model.predicting_ids, model.cost, 
                                                 model.optimizer, model.accuracy], 
                                      feed_dict={model.X:batch_x,
                                                model.Y:batch_y})
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= (len(short_questions) / batch_size)
    total_accuracy /= (len(short_questions) / batch_size)
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))


# In[ ]:


for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')


# In[ ]:


batch_x, _ = pad_sentence_batch_static(X_test[:batch_size], PAD)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD)
predicted = sess.run(model.predicting_ids, feed_dict={model.X:batch_x})

for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')

