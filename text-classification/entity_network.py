# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib import rnn


class EntityNetwork:
    def __init__(
        self,
        num_classes,
        learning_rate,
        decay_steps,
        decay_rate,
        sequence_length,
        story_length,
        vocab_size,
        embed_size,
        hidden_size,
        block_size = 20,
        initializer = tf.random_normal_initializer(stddev = 0.1),
        clip_gradients = 5.0,
        use_bi_lstm = False,
    ):
        """init all hyperparameter here"""
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(
            learning_rate, trainable = False, name = 'learning_rate'
        )
        self.learning_rate_decay_half_op = tf.assign(
            self.learning_rate, self.learning_rate * 0.5
        )
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients
        self.story_length = story_length
        self.block_size = block_size
        self.use_bi_lstm = use_bi_lstm
        self.dimension = (
            self.hidden_size * 2 if self.use_bi_lstm else self.hidden_size
        )
        self.story = tf.placeholder(
            tf.int32,
            [None, self.story_length, self.sequence_length],
            name = 'story',
        )
        self.query = tf.placeholder(
            tf.int32, [None, self.sequence_length], name = 'question'
        )
        self.batch_size = tf.shape(self.query)[0]
        self.answer_single = tf.placeholder(tf.int32, [None], name = 'input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name = 'dropout_keep_prob'
        )

        self.global_step = tf.Variable(
            0, trainable = False, name = 'Global_Step'
        )
        self.epoch_step = tf.Variable(0, trainable = False, name = 'Epoch_Step')
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1))
        )
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        self.predictions = tf.argmax(self.logits, 1, name = 'predictions')
        correct_prediction = tf.equal(
            tf.cast(self.predictions, tf.int32), self.answer_single
        )
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name = 'Accuracy'
        )
        self.cost = self.loss()
        self.optimizer = self.train()

    def inference(self):
        self.embedding_with_mask()
        if self.use_bi_lstm:
            self.input_encoder_bi_lstm()
        else:
            self.input_encoder_bow()
        self.hidden_state = self.rnn_story()
        logits = self.output_module()
        return logits

    def output_module(self):
        p = tf.nn.softmax(
            tf.multiply(
                tf.expand_dims(self.query_embedding, axis = 1),
                self.hidden_state,
            )
        )
        u = tf.reduce_sum(tf.multiply(p, self.hidden_state), axis = 1)
        H_u_matmul = tf.matmul(u, self.H) + self.h_u_bias
        activation = self.activation(
            self.query_embedding + H_u_matmul, scope = 'query_add_hidden'
        )
        activation = tf.nn.dropout(
            activation, keep_prob = self.dropout_keep_prob
        )
        y = tf.matmul(activation, self.R) + self.y_bias
        return y

    def rnn_story(self):
        input_split = tf.split(
            self.story_embedding, self.story_length, axis = 1
        )
        input_list = [tf.squeeze(x, axis = 1) for x in input_split]
        h_all = tf.get_variable(
            'hidden_states',
            shape = [self.block_size, self.dimension],
            initializer = self.initializer,
        )
        w_all = tf.get_variable(
            'keys',
            shape = [self.block_size, self.dimension],
            initializer = self.initializer,
        )
        w_all_expand = tf.tile(
            tf.expand_dims(w_all, axis = 0), [self.batch_size, 1, 1]
        )
        h_all_expand = tf.tile(
            tf.expand_dims(h_all, axis = 0), [self.batch_size, 1, 1]
        )
        for i, input in enumerate(input_list):
            h_all_expand = self.cell(input, h_all_expand, w_all_expand, i)
        return h_all_expand

    def embedding_with_mask(self):
        story_embedding = tf.nn.embedding_lookup(self.Embedding, self.story)
        query_embedding = tf.nn.embedding_lookup(self.Embedding, self.query)
        story_mask = tf.get_variable(
            'story_mask',
            [self.sequence_length, 1],
            initializer = tf.constant_initializer(1.0),
        )
        query_mask = tf.get_variable(
            'query_mask',
            [self.sequence_length, 1],
            initializer = tf.constant_initializer(1.0),
        )
        self.story_embedding = tf.multiply(story_embedding, story_mask)
        self.query_embedding = tf.multiply(query_embedding, query_mask)

    def input_encoder_bow(self):
        self.story_embedding = tf.reduce_sum(self.story_embedding, axis = 2)
        self.query_embedding = tf.reduce_sum(self.query_embedding, axis = 1)

    def input_encoder_bi_lstm(self):
        """
        use bi-directional lstm to encode query_embedding:[batch_size,sequence_length,embed_size]
        and story_embedding:[batch_size,story_length,sequence_length,embed_size]
        output:query_embedding:[batch_size,hidden_size*2]
        story_embedding:[batch_size,self.story_length,self.hidden_size*2]
        """
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(
                lstm_fw_cell, output_keep_prob = self.dropout_keep_prob
            )
            lstm_bw_cell == rnn.DropoutWrapper(
                lstm_bw_cell, output_keep_prob = self.dropout_keep_prob
            )
        query_hidden_output, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell,
            lstm_bw_cell,
            self.query_embedding,
            dtype = tf.float32,
            scope = 'query_rnn',
        )
        query_hidden_output = tf.concat(query_hidden_output, axis = 2)
        self.query_embedding = tf.reduce_sum(query_hidden_output, axis = 1)
        self.story_embedding = tf.reshape(
            self.story_embedding,
            shape = (
                -1,
                self.story_length * self.sequence_length,
                self.embed_size,
            ),
        )
        lstm_fw_cell_story = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell_story = rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell_story = rnn.DropoutWrapper(
                lstm_fw_cell_story, output_keep_prob = self.dropout_keep_prob
            )

    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope('output_module'):
            self.H = tf.get_variable(
                'H',
                shape = [self.dimension, self.dimension],
                initializer = self.initializer,
            )
            self.R = tf.get_variable(
                'R',
                shape = [self.dimension, self.num_classes],
                initializer = self.initializer,
            )
            self.y_bias = tf.get_variable('y_bias', shape = [self.num_classes])
            self.b_projected = tf.get_variable(
                'b_projection', shape = [self.num_classes]
            )
            self.h_u_bias = tf.get_variable(
                'h_u_bias', shape = [self.dimension]
            )

        with tf.variable_scope('dynamic_memory'):
            self.U = tf.get_variable(
                'U',
                shape = [self.dimension, self.dimension],
                initializer = self.initializer,
            )
            self.V = tf.get_variable(
                'V',
                shape = [self.dimension, self.dimension],
                initializer = self.initializer,
            )
            self.W = tf.get_variable(
                'W',
                shape = [self.dimension, self.dimension],
                initializer = self.initializer,
            )
            self.h_bias = tf.get_variable('h_bias', shape = [self.dimension])
            self.h2_bias = tf.get_variable('h2_bias', shape = [self.dimension])

        with tf.variable_scope('embedding_projection'):
            self.Embedding = tf.get_variable(
                'Embedding',
                shape = [self.vocab_size, self.embed_size],
                initializer = self.initializer,
            )

    def cell(self, s_t, h_all, w_all, i):
        s_t_expand = tf.expand_dims(s_t, axis = 1)
        g = tf.nn.sigmoid(
            tf.multiply(s_t_expand, h_all) + tf.multiply(s_t_expand, w_all)
        )

        h_candidate_part1 = (
            tf.matmul(tf.reshape(h_all, shape = (-1, self.dimension)), self.U)
            + tf.matmul(tf.reshape(w_all, shape = (-1, self.dimension)), self.V)
            + self.h_bias
        )

        h_candidate_part1 = tf.reshape(
            h_candidate_part1,
            shape = (self.batch_size, self.block_size, self.dimension),
        )
        h_candidate_part2 = tf.expand_dims(
            tf.matmul(s_t, self.W) + self.h2_bias, axis = 1
        )
        h_candidate = self.activation(
            h_candidate_part1 + h_candidate_part2,
            scope = 'h_candidate' + str(i),
        )

        h_all = h_all + tf.multiply(g, h_candidate)

        h_all = tf.nn.l2_normalize(h_all, -1)
        return h_all

    def activation(self, features, scope = None):
        with tf.variable_scope(scope, 'PReLU', initializer = self.initializer):
            alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
            pos = tf.nn.relu(features)
            neg = alpha * (features - tf.abs(features)) * 0.5
            return pos + neg

    def loss(self, l2_lambda = 0.0001):  # 0.001
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.answer_single, logits = self.logits
            )
            loss = tf.reduce_mean(losses)
            l2_losses = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if ('bias' not in v.name) and ('alpha' not in v.name)
                    ]
                )
                * l2_lambda
            )
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase = True,
        )
        self.learning_rate_ = learning_rate
        train_op = tf_contrib.layers.optimize_loss(
            self.cost,
            global_step = self.global_step,
            learning_rate = learning_rate,
            optimizer = 'Adam',
            clip_gradients = self.clip_gradients,
        )
        return train_op
