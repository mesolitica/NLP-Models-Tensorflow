from setting import (
    embed_size,
    char2idx,
    learning_rate,
    encoder_num_banks,
    n_mels,
    reduction_factor,
    num_highway_blocks,
)
from modules import (
    conv1d,
    normalize_in,
    highwaynet,
    gru,
    attention_decoder,
    prenet,
    embed,
    shift_by_one,
    conv1d_banks,
)
import tensorflow as tf


def encode(inputs, is_training = True, scope = 'encoder', reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        prenet_out = prenet(inputs, scope = 'prenet', is_training = is_training)
        enc = conv1d_banks(
            prenet_out, K = encoder_num_banks, is_training = is_training
        )
        enc = tf.layers.max_pooling1d(enc, 2, 1, padding = 'same')
        enc = conv1d(enc, embed_size // 2, 3, scope = 'conv1d_1')
        enc = normalize_in(enc, activation_fn = tf.nn.relu)
        enc = conv1d(enc, embed_size // 2, 3, scope = 'conv1d_2')
        enc = normalize_in(enc, activation_fn = tf.nn.relu)
        enc += prenet_out
        for i in range(num_highway_blocks):
            enc = highwaynet(
                enc, units = embed_size // 2, scope = 'highwaynet_%d' % (i)
            )
        memory = gru(enc, embed_size // 2, True)
    return memory


def decode(
    inputs, memory, is_training = True, scope = 'decoder_layers', reuse = None
):
    with tf.variable_scope(scope, reuse = reuse):
        dec = prenet(inputs, is_training = is_training)
        dec = attention_decoder(dec, memory, embed_size)
        dec += gru(dec, embed_size, False, scope = 'gru1')
        dec += gru(dec, embed_size, False, scope = 'gru2')
        return tf.layers.dense(dec, len(char2idx))


class Model:
    def __init__(self, is_training = True):
        self.X = tf.placeholder(
            tf.float32, shape = (None, None, n_mels * reduction_factor)
        )
        self.Y = tf.placeholder(tf.int32, shape = (None, None))
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.decoder_inputs = embed(
            shift_by_one(self.Y), len(char2idx), embed_size
        )
        with tf.variable_scope('net'):
            self.memory = encode(self.X, is_training = is_training)
            self.outputs = decode(
                self.decoder_inputs, self.memory, is_training = is_training
            )
            self.logprobs = tf.log(tf.nn.softmax(self.outputs) + 1e-10)
            self.preds = tf.argmax(self.outputs, axis = -1)
            correct_pred = tf.equal(tf.cast(self.preds, tf.int32), self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        if is_training:
            masks = tf.sequence_mask(
                self.Y_seq_len,
                tf.reduce_max(self.Y_seq_len),
                dtype = tf.float32,
            )
            self.cost = tf.contrib.seq2seq.sequence_loss(
                logits = self.outputs, targets = self.Y, weights = masks
            )
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.cost
            )
