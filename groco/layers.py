from tensorflow.keras.layers import Conv2D
import tensorflow as tf
from functools import partial
from groco import groups
from groco.utils import EquivariantPadding


class GroupConv2D(Conv2D):
    """
    Group convolution layer built on Keras's Conv2D layer.
    Additional arguments:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    Expected input shape: (batch, height, width, group_order, in_channels), with the group_order axis omitted for the
    first layer (the lifting convolution).

    NOTE: the option data_format='channels_first' is not supported.
    """

    def __init__(self, group, kernel_size, allow_non_equivariance: bool=False, **kwargs):
        self.group = group if isinstance(group, groups.WallpaperGroup) else groups.group_dict[group]

        self.equivariant_padding = EquivariantPadding(allow_non_equivariance=allow_non_equivariance,
                                                      kernel_size=kernel_size, **kwargs)
        if 'padding' in kwargs and kwargs['padding'].endswith('_equiv'):
            kwargs['padding'] = kwargs['padding'][:-6]

        super().__init__(kernel_size=kernel_size, **kwargs)

        # set during build
        self.group_valued_input = None
        self.repeated_bias_indices = None
        self.transformed_kernel_indices = None

    def call(self, inputs):
        group_merged_inputs = self._merge_group_axis(inputs)
        padded_group_merged_inputs = self.equivariant_padding(group_merged_inputs)

        outputs = self._conv2d_call(
            padded_group_merged_inputs,
            self._transformed_kernel(),
            self._repeated_bias()
        )
        return self._restore_group_axis(outputs)

    def _merge_group_axis(self, inputs):
        """
        If the input is a signal on the group, join the group axis with the channel axis.

        (batch, height, width, group_order, channels) -> (batch, height, width, group_order * channels)
        """
        if not self.group_valued_input:
            return inputs

        batch, height, width, group_order, channels = inputs.shape
        batch = -1 if batch is None else batch
        return tf.reshape(inputs, (batch, height, width, group_order * channels))

    def _transformed_kernel(self):
        """Perform the group action on the kernel, as a signal on the group, or on the grid in a lifting convolution."""
        return tf.gather(self.kernel, indices=self.transformed_kernel_indices, axis=0)

    def _repeated_bias(self):
        """Transform the bias to group.order repeated copies of itself."""
        return tf.gather(self.bias, indices=self.repeated_bias_indices, axis=0)

    def _restore_group_axis(self, outputs):
        """
        Reshape the output of the convolution, splitting off the group index from the channel axis.

        (batch, height, width, group_order * channels) -> (batch, height, width, group_order, channels)
        """
        batch, height, width, channels = outputs.shape
        batch = -1 if batch is None else batch
        return tf.reshape(outputs, (batch, height, width, self.group.order, channels // self.group.order))

    def build(self, input_shape):
        """
        Check if the input is a signal on the group (rather than just on the grid),
        and stores that in attribute group_valued_input.
        If so, merge the group axis with the channel axis.
        Then run the parent class's build.
        Run the EquivariantPadding layer's build on the merged input.
        Construct repeated_bias_indices and transformed_kernel_indices
        """
        self.group_valued_input = len(input_shape) == 5  # this includes the batch dimension
        if self.group_valued_input:
            (batch, height, width, group_order, channels) = input_shape
            assert group_order == self.group.order, f'Got input shape {input_shape}, expected {(batch, height, width, self.group.order, channels)}.'
            input_shape = (batch, height, width, channels * group_order)

        super().build(input_shape)

        if self.group_valued_input:
            self.input_spec.axes = {self._get_channel_axis(): channels}

        self.equivariant_padding.build(input_shape)

        self.repeated_bias_indices = self._compute_repeated_bias_indices()
        self.transformed_kernel_indices = self._compute_transformed_kernel_indices()

        self.kernel = tf.reshape(self.kernel, [-1])

    def _compute_repeated_bias_indices(self):
        """Compute a 1D tensor of indices used to gather from the bias in order to repeat it across the group axis."""
        indices = tf.range(tf.size(self.bias))
        indices = tf.concat([indices for _ in range(self.group.order)], axis=0)
        return indices

    def _compute_transformed_kernel_indices(self):
        """Compute a tensor of indices used to gather from the kernel to produce the group action on it."""
        indices = tf.reshape(tf.range(tf.size(self.kernel)), self.kernel.shape)
        indices = self.group.action(indices)
        if self.group_valued_input:
            indices = self._permute_group_axis(indices)
        indices = self._merge_group_channels_out(indices)
        return indices

    def _permute_group_axis(self, kernel):
        """
        After the action on the spatial part has been done, perform the action on the group argument.
        Permute the input group axis according to the group composition.

        (height, width, group_order, group_order * channels_in, channels_out) -> same
        """
        (height, width, group_order, channels_in, channels_out) = kernel.shape

        transformed_kernel = tf.reshape(kernel, (height, width, group_order * group_order, channels_in // group_order, channels_out))
        transformed_kernel = tf.gather(transformed_kernel, axis=-3, indices=self.group.composition_indices)
        transformed_kernel = tf.reshape(transformed_kernel, (height, width, group_order, channels_in, channels_out))
        return transformed_kernel

    @staticmethod
    def _merge_group_channels_out(kernel):
        """
        Merge the group axis with the channels_out axis.

        (height, width, group_order, channels_in, channels_out) -> (height, width, channels_in, group_order * channels_out)
        """
        height, width, group_order, channels_in, channels_out = kernel.shape
        transformed_kernel = tf.transpose(kernel, (0, 1, 3, 2, 4))
        return tf.reshape(transformed_kernel, (height, width, channels_in, group_order * channels_out))

    def group_transform(self, signal, group_element=None):
        """
        Act with the whole group on a signal, which can be a signal on the domain or the group.
        Used only as a helper method to check equivariance, not internally.

        :param signal: Tensor of shape (batch, height, width, group_order, channels), group_order possibly omitted.
        :param group_element: Index of group element to act with. Defaults to None meaning all group elements are used.
        :return: transformed tensor of shape (group_order, batch, height, width, group_order, channels).
        The first axis indicates which element was acted with, and is omitted if a group_element is specified.
        """
        group_valued_signal = signal.shape.rank == 5

        # reshape to match the kernel
        if group_valued_signal:
            (batch, height, width, group_order, channels) = signal.shape
            assert group_order == self.group.order, f'Got input shape {signal.shape}, expected {(batch, height, width, self.group.order, channels)}.'
            transformed_signal = tf.transpose(signal, (1, 2, 3, 4, 0))
            transformed_signal = tf.reshape(transformed_signal, (height, width, group_order * channels, batch))
        else:
            transformed_signal = tf.transpose(signal, (1, 2, 3, 0))
        # shape now (height, width, channels, batch)

        transformed_signal = self.group.action(transformed_signal)
        # shape now (height, width, group_order, channel, batch)

        if group_valued_signal:
            transformed_signal = self._permute_group_axis(transformed_signal)

        transformed_signal = tf.transpose(transformed_signal, (2, 4, 0, 1, 3))
        # shape now (group_order, batch, height, width, channel)
        if group_valued_signal:
          transformed_signal = tf.reshape(transformed_signal, (group_order, batch, height, width, group_order, channels))

        if group_element is not None:
          transformed_signal = transformed_signal[group_element]

        return transformed_signal

    def _conv2d_call(self, reshaped_inputs, reshaped_kernel, reshaped_bias):
        """
        This is the original call method, with replacements:
        - self.kernel -> reshaped_kernel (argument)
        - self.bias -> reshaped_bias (argument)
        - self.filters -> filters_multiplied = self.filters * self.group_order
        """
        input_shape = reshaped_inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            reshaped_inputs = tf.pad(reshaped_inputs, self._compute_causal_padding(reshaped_inputs))

        outputs = self._convolution_op(reshaped_inputs, reshaped_kernel)  # had to add underscore to make it work

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                filters_multiplied = self.filters * self.group.order
                bias = tf.reshape(reshaped_bias, (1, filters_multiplied, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(o, reshaped_bias, data_format=self._tf_data_format)

                    outputs = tf.keras.utils.conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = tf.nn.bias_add(
                        outputs, reshaped_bias, data_format=self._tf_data_format)

        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self._compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        """
        Multiply channels with group order.
        Note the layer's output shape is changed later to have a separate group index, this is to be
        consistent with the conv2d_call method.
        """
        out_shape = super().compute_output_shape(input_shape)
        batch, height, width, channels = out_shape
        out_shape = batch, height, width, channels * self.group.order
        return tf.TensorShape(out_shape)


P4MConv2D = partial(GroupConv2D, group=groups.P4M)
P4Conv2D = partial(GroupConv2D, group=groups.P4)
P2MMConv2D = partial(GroupConv2D, group=groups.P2MM)
PMhConv2D = partial(GroupConv2D, group=groups.PMh)
PMwConv2D = partial(GroupConv2D, group=groups.PMw)
P2Conv2D = partial(GroupConv2D, group=groups.P2)
P1Conv2D = partial(GroupConv2D, group=groups.P1)
