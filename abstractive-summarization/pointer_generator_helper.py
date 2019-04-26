import tensorflow as tf
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import (
    _compute_attention,
)

UNK_ID = 3


class PointerGeneratorGreedyEmbeddingHelper(
    tf.contrib.seq2seq.GreedyEmbeddingHelper
):
    def __init__(self, embedding, start_tokens, end_token):
        self.vocab_size = tf.shape(embedding)[-1]
        super(PointerGeneratorGreedyEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token
        )

    def sample(self, time, outputs, state, name = None):
        """sample for PointerGeneratorGreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError(
                'Expected outputs to be a single Tensor, got: %s'
                % type(outputs)
            )
        sample_ids = tf.argmax(outputs, axis = -1, output_type = tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name = None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)

        # since we have OOV words, we need change these words to UNK
        condition = tf.less(sample_ids, self.vocab_size)
        sample_ids = tf.where(
            condition, sample_ids, tf.ones_like(sample_ids) * UNK_ID
        )

        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids),
        )
        return (finished, next_inputs, state)


class PointerGeneratorDecoder(tf.contrib.seq2seq.BasicDecoder):
    """Pointer Generator sampling decoder."""

    def __init__(
        self,
        source_extend_tokens,
        source_oov_words,
        coverage,
        cell,
        helper,
        initial_state,
        output_layer = None,
    ):
        self.source_oov_words = source_oov_words
        self.source_extend_tokens = source_extend_tokens
        self.coverage = coverage
        super(PointerGeneratorDecoder, self).__init__(
            cell, helper, initial_state, output_layer
        )

    @property
    def output_size(self):
        # Return the cell output and the id
        return tf.contrib.seq2seq.BasicDecoderOutput(
            rnn_output = self._rnn_output_size() + self.source_oov_words,
            sample_id = self._helper.sample_ids_shape,
        )

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return tf.contrib.seq2seq.BasicDecoderOutput(
            nest.map_structure(
                lambda _: dtype, self._rnn_output_size() + self.source_oov_words
            ),
            self._helper.sample_ids_dtype,
        )

    def step(self, time, inputs, state, name = None):
        """Perform a decoding step.
        Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.
        Returns:
        `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, 'PGDecoderStep', (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            # the first cell state contains attention, which is context
            attention = cell_state[0].attention
            att_cell_state = cell_state[0].cell_state
            alignments = cell_state[0].alignments

            with tf.variable_scope('calculate_pgen'):
                p_gen = _linear([attention, inputs, att_cell_state], 1, True)
                p_gen = tf.sigmoid(p_gen)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            vocab_dist = tf.nn.softmax(cell_outputs) * p_gen

            # z = tf.reduce_sum(alignments,axis=1)
            # z = tf.reduce_sum(tf.cast(tf.less_equal(alignments, 0),tf.int32))
            alignments = alignments * (1 - p_gen)

            # x = tf.reduce_sum(tf.cast(tf.less_equal((1-p_gen), 0),tf.int32))
            # y = tf.reduce_sum(tf.cast(tf.less_equal(alignments[3], 0),tf.int32))

            # this is only for debug
            # alignments2 =  tf.Print(alignments2,[tf.shape(inputs),x,y,alignments[2][9:12]],message="zeros in vocab dist and alignments")

            # since we have OOV words, we need expand the vocab dist
            vocab_size = tf.shape(vocab_dist)[-1]
            extended_vsize = vocab_size + self.source_oov_words
            batch_size = tf.shape(vocab_dist)[0]
            extra_zeros = tf.zeros((batch_size, self.source_oov_words))
            # batch * extend vocab size
            vocab_dists_extended = tf.concat(
                axis = -1, values = [vocab_dist, extra_zeros]
            )
            # vocab_dists_extended = tf.Print(vocab_dists_extended,[tf.shape(vocab_dists_extended),self.source_oov_words],message='vocab_dists_extended size')

            batch_nums = tf.range(0, limit = batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self.source_extend_tokens)[
                1
            ]  # number of states we attend over
            batch_nums = tf.tile(
                batch_nums, [1, attn_len]
            )  # shape (batch_size, attn_len)
            indices = tf.stack(
                (batch_nums, self.source_extend_tokens), axis = 2
            )  # shape (batch_size, enc_t, 2)
            shape = [batch_size, extended_vsize]
            attn_dists_projected = tf.scatter_nd(indices, alignments, shape)

            final_dists = attn_dists_projected + vocab_dists_extended
            # final_dists = tf.Print(final_dists,[tf.reduce_sum(tf.cast(tf.less_equal(final_dists[0],0),tf.int32))],message='final dist')
            # note: sample_ids will contains OOV words
            sample_ids = self._helper.sample(
                time = time, outputs = final_dists, state = cell_state
            )

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time = time,
                outputs = cell_outputs,
                state = cell_state,
                sample_ids = sample_ids,
            )

            outputs = tf.contrib.seq2seq.BasicDecoderOutput(
                final_dists, sample_ids
            )
            return (outputs, next_state, next_inputs, finished)


class PointerGeneratorAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    def __init__(
        self,
        cell,
        attention_mechanism,
        attention_layer_size = None,
        alignment_history = False,
        cell_input_fn = None,
        output_attention = True,
        initial_cell_state = None,
        name = None,
        coverage = False,
    ):
        super(PointerGeneratorAttentionWrapper, self).__init__(
            cell,
            attention_mechanism,
            attention_layer_size,
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name,
        )
        self.coverage = coverage

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
        batch_size: `0D` integer tensor: the batch size.
        dtype: The internal state data type.
        Returns:
        An `AttentionWrapperState` tuple containing zeroed out tensors and,
        possibly, empty `TensorArray` objects.
        Raises:
        ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(
            type(self).__name__ + 'ZeroState', values = [batch_size]
        ):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                'When calling zero_state of AttentionWrapper %s: '
                % self._base_name
                + 'Non-matching batch sizes between the memory '
                '(encoder output) and the requested batch size.  Are you using '
                'the BeamSearchDecoder?  If so, make sure your encoder output has '
                'been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and '
                'the batch_size= argument passed to zero_state is '
                'batch_size * beam_width.'
            )
            with tf.control_dependencies(
                self._batch_size_checks(batch_size, error_message)
            ):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name = 'checked_cell_state'),
                    cell_state,
                )
            return tf.contrib.seq2seq.AttentionWrapperState(
                cell_state = cell_state,
                time = tf.zeros([], dtype = tf.int32),
                attention = _zero_state_tensors(
                    self._attention_layer_size, batch_size, dtype
                ),
                alignments = self._item_or_tuple(
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms
                ),
                attention_state = self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms
                ),
                # since we need to read the alignment history several times, so we need set clear_after_read to False
                alignment_history = self._item_or_tuple(
                    tf.TensorArray(
                        dtype = dtype,
                        size = 0,
                        clear_after_read = False,
                        dynamic_size = True,
                    )
                    if self._alignment_history
                    else ()
                    for _ in self._attention_mechanisms
                ),
            )

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).
        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                containing the state calculated at this time step.
        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError(
                'Expected state to be instance of AttentionWrapperState. '
                'Received type %s instead.' % type(state)
            )

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = cell_output.shape[0].value or tf.shape(cell_output)[0]
        error_message = (
            'When applying AttentionWrapper %s: ' % self.name
            + 'Non-matching batch sizes between the memory '
            '(encoder output) and the query (decoder output).  Are you using '
            'the BeamSearchDecoder?  You may need to tile your memory input via '
            'the tf.contrib.seq2seq.tile_batch function with argument '
            'multiple=beam_width.'
        )
        with tf.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)
        ):
            cell_output = tf.identity(cell_output, name = 'checked_cell_output')

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            print(attention_mechanism)
            if self.coverage:
                # if we use coverage mode, previous alignments is coverage vector
                # alignment history stack has shape:  decoder time * batch * atten_len
                # convert it to coverage vector
                previous_alignments[i] = tf.cond(
                    previous_alignment_history[i].size() > 0,
                    lambda: tf.reduce_sum(
                        tf.transpose(
                            previous_alignment_history[i].stack(), [1, 2, 0]
                        ),
                        axis = 2,
                    ),
                    lambda: tf.zeros_like(previous_alignments[i]),
                )
            # debug
            # previous_alignments[i] = tf.Print(previous_alignments[i],[previous_alignment_history[i].size(), tf.shape(previous_alignments[i]),previous_alignments[i]],message="atten wrapper:")
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism,
                cell_output,
                previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None,
            )
            alignment_history = (
                previous_alignment_history[i].write(state.time, alignments)
                if self._alignment_history
                else ()
            )

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        attention = tf.concat(all_attentions, 1)
        next_state = tf.contrib.seq2seq.AttentionWrapperState(
            time = state.time + 1,
            cell_state = next_cell_state,
            attention = attention,
            alignments = self._item_or_tuple(all_alignments),
            attention_state = self._item_or_tuple(all_alignments),
            alignment_history = self._item_or_tuple(all_histories),
        )

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


def _pg_bahdanau_score(processed_query, keys, coverage, coverage_vector):
    """Implements Bahdanau-style (additive) scoring function.
    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        coverage: Whether to use coverage mode.
        coverage_vector: only used when coverage is true
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    v = tf.get_variable('attention_v', [num_units], dtype = dtype)
    b = tf.get_variable(
        'attention_b',
        [num_units],
        dtype = dtype,
        initializer = tf.zeros_initializer(),
    )
    if coverage:
        w_c = tf.get_variable('coverage_w', [num_units], dtype = dtype)
        # debug
        # coverage_vector = tf.Print(coverage_vector,[coverage_vector],message="score")
        coverage_vector = tf.expand_dims(coverage_vector, -1)
        return tf.reduce_sum(
            v * tf.tanh(keys + processed_query + coverage_vector * w_c + b), [2]
        )
    else:
        return tf.reduce_sum(v * tf.tanh(keys + processed_query + b), [2])


class PointerGeneratorBahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
    def __init__(
        self,
        num_units,
        memory,
        memory_sequence_length = None,
        normalize = False,
        probability_fn = None,
        score_mask_value = float('-inf'),
        name = 'PointerGeneratorBahdanauAttention',
        coverage = False,
    ):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
            name: Name to use when creating ops.
            coverage: whether use coverage mode
        """
        super(PointerGeneratorBahdanauAttention, self).__init__(
            num_units = num_units,
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            normalize = normalize,
            probability_fn = probability_fn,
            score_mask_value = score_mask_value,
            name = name,
        )
        self.coverage = coverage

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with tf.variable_scope(
            None, 'pointer_generator_bahdanau_attention', [query]
        ):
            processed_query = (
                self.query_layer(query) if self.query_layer else query
            )
            score = _pg_bahdanau_score(
                processed_query, self._keys, self.coverage, state
            )
        # Note: state is not used in probability_fn in Bahda attention, so I use it as coverage vector in coverage mode
        alignments = self._probability_fn(score, state)
        next_state = alignments
        print(alignments, next_state)
        return alignments, next_state
