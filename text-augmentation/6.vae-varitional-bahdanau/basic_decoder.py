import collections

import tensorflow as tf

import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
]


class BasicDecoderOutput(collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
    pass


class BasicDecoder(decoder.Decoder):
    """Basic sampling decoder."""

    def __init__(self, cell, helper, initial_state, latent_vector, output_layer=None):
        """Initialize BasicDecoder.
        Args:
          cell: An `RNNCell` instance.
          helper: A `Helper` instance.
          initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
            The initial state of the RNNCell.
          output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
            to storing the result or sampling.
        Raises:
          TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
        """
        if not isinstance(helper, helper_py.Helper):
            raise TypeError("helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError("output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self._latent_vector = latent_vector
        self._intermediate_context_kl_loss = tf.zeros(shape=(helper.batch_size,))  # shape of (batch_size,)
        # CHANGE-1: Variable to keep adding the c_kl_losses from each timestep

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer.compute_output_shape(  # pylint: disable=protected-access
                output_shape_with_unknown_batch)
        return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and int32 (the id)
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            dtypes.int32)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
          name: Name scope for any created operations.
        Returns:
          `(finished, first_inputs, initial_state)`.
        """
        # Concatenate the latent vector to the 1st input to the decoder LSTM, i.e, the <GO> embedding + latent vector
        return (self._helper.initialize()[0],
                tf.concat([self._helper.initialize()[1], self._latent_vector], axis=-1)) + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.
        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state, c_kl_loss = self._cell(inputs, state)
            # Accumulate the context KL loss from token at the current decoder step
            self._intermediate_context_kl_loss += c_kl_loss
            c_kl_loss = self._intermediate_context_kl_loss

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)

            # Concatenate the latent vector to the predicted word's embedding
            next_inputs = tf.concat([next_inputs, self._latent_vector], axis=-1)

        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished, c_kl_loss)