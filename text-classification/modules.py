from setting import embed_size
import tensorflow as tf


def embed(inputs, vocab_size, dimension, scope = 'embedding', reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        lookup_table = tf.get_variable(
            'lookup_table',
            dtype = tf.float32,
            shape = [vocab_size, dimension],
            initializer = tf.truncated_normal_initializer(
                mean = 0.0, stddev = 0.01
            ),
        )
        lookup_table = tf.concat(
            (tf.zeros(shape = [1, dimension]), lookup_table[1:, :]), 0
        )
    return tf.nn.embedding_lookup(lookup_table, inputs)


def normalize_bn(
    inputs,
    decay = 0.99,
    is_training = True,
    activation_fn = None,
    scope = 'normalize_bn',
):
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis = 1)
            inputs = tf.expand_dims(inputs, axis = 2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis = 1)
        outputs = tf.contrib.layers.batch_norm(
            inputs = inputs,
            decay = decay,
            center = True,
            scale = True,
            activation_fn = activation_fn,
            updates_collections = None,
            is_training = is_training,
            scope = scope,
            zero_debias_moving_mean = True,
            fused = True,
        )
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis = [1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis = 1)
    else:
        outputs = tf.contrib.layers.batch_norm(
            inputs = inputs,
            decay = decay,
            center = True,
            scale = True,
            activation_fn = activation_fn,
            updates_collections = None,
            is_training = is_training,
            scope = scope,
            fused = False,
        )
    return outputs


def normalize_layer_norm(
    inputs, activation_fn = None, scope = 'normalize_layer_norm'
):
    return tf.contrib.layers.layer_norm(
        inputs = inputs,
        center = True,
        scale = True,
        activation_fn = activation_fn,
        scope = scope,
    )


def normalize_in(inputs, activation_fn = None, scope = 'normalize_in'):
    with tf.variable_scope(scope):
        batch, steps, channels = inputs.get_shape().as_list()
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(inputs, [1], keep_dims = True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-8
        normalized = (inputs - mu) / (sigma_sq + epsilon) ** (0.5)
        outputs = scale * normalized + shift
        if activation_fn:
            outputs = activation_fn(outputs)
    return outputs


def conv1d(
    inputs,
    filters = None,
    size = 1,
    rate = 1,
    padding = 'SAME',
    use_bias = False,
    activation_fn = None,
    scope = 'conv1d',
    reuse = None,
):
    with tf.variable_scope(scope):
        if padding.lower() == 'causal':
            pad_len = (size - 1) * rate
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = 'valid'
        if filters is None:
            filters = inputs.get_shape().as_list()[-1]
        params = {
            'inputs': inputs,
            'filters': filters,
            'kernel_size': size,
            'dilation_rate': rate,
            'padding': padding,
            'activation': activation_fn,
            'use_bias': use_bias,
            'reuse': reuse,
        }
        outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(
    inputs, K = 16, is_training = True, scope = 'conv1d_banks', reuse = None
):
    with tf.variable_scope(scope, reuse = reuse):
        outputs = conv1d(inputs, embed_size // 2, 1)
        outputs = normalize_in(outputs, tf.nn.relu)
        for k in range(2, K + 1):
            with tf.variable_scope('num_%d' % (k)):
                output = conv1d(inputs, embed_size // 2, k)
                output = normalize_in(output, tf.nn.relu)
                outputs = tf.concat((outputs, output), -1)
    return outputs


def gru(inputs, units = None, bidirection = False, scope = 'gru', reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        if units is None:
            units = inputs.get_shape().as_list()[-1]
        cell = tf.contrib.rnn.GRUCell(units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell, cell_bw, inputs, dtype = tf.float32
            )
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
            return outputs


def attention_decoder(
    inputs, memory, units = None, scope = 'attention_decoder', reuse = None
):
    with tf.variable_scope(scope, reuse = reuse):
        if units is None:
            units = inputs.get_shape().as_list()[-1]
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            units, memory
        )
        decoder_cell = tf.contrib.rnn.GRUCell(units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism, units
        )
        outputs, _ = tf.nn.dynamic_rnn(
            cell_with_attention, inputs, dtype = tf.float32
        )
    return outputs


def prenet(inputs, is_training = True, scope = 'prenet', reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        outputs = tf.layers.dense(
            inputs, units = embed_size, activation = tf.nn.relu, name = 'dense1'
        )
        outputs = tf.nn.dropout(
            outputs,
            keep_prob = 0.5 if is_training == True else 1.0,
            name = 'dropout1',
        )
        outputs = tf.layers.dense(
            outputs,
            units = embed_size // 2,
            activation = tf.nn.relu,
            name = 'dense2',
        )
        outputs = tf.nn.dropout(
            outputs,
            keep_prob = 0.5 if is_training == True else 1.0,
            name = 'dropout2',
        )
    return outputs


def highwaynet(inputs, units = None, scope = 'highwaynet', reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        if units is None:
            units = inputs.get_shape().as_list()[-1]
        H = tf.layers.dense(
            inputs, units = units, activation = tf.nn.relu, name = 'dense1'
        )
        T = tf.layers.dense(
            inputs, units = units, activation = tf.nn.sigmoid, name = 'dense2'
        )
        C = 1.0 - T
        return H * T + inputs * C


def shift_by_one(inputs):
    return tf.concat((tf.zeros_like(inputs[:, :1]), inputs[:, :-1]), 1)
