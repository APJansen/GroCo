from tensorflow.keras.layers import Conv2D
import tensorflow as tf
from functools import partial
from groco.groups import Group, wallpaper_groups
from groco.layers import EquivariantPadding


class GroupConv2D(Conv2D):
    """
    Group convolution layer built on Keras's Conv2D layer.
    Additional arguments:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    Expected input shape: (batch, height, width, group.order, in_channels), with the group.order axis omitted for the
    first layer (the lifting convolution).

    """

    def __init__(self, group, kernel_size, dimensions=2, allow_non_equivariance: bool = False, subgroup=None, **kwargs):
        self.group = group if isinstance(group, Group) else wallpaper_groups.group_dict[group]
        self.subgroup_name = subgroup
        self.subgroup = self.group if subgroup is None else wallpaper_groups.group_dict[subgroup]
        self.dimensions = dimensions

        self.equivariant_padding = EquivariantPadding(
            allow_non_equivariance=allow_non_equivariance, kernel_size=kernel_size, dimensions=2, **kwargs)
        if 'padding' in kwargs and kwargs['padding'].endswith('_equiv'):
            kwargs['padding'] = kwargs['padding'][:-6]

        super().__init__(kernel_size=kernel_size, **kwargs)

        # axes refer to input
        self.channels_axis = 1 if self.data_format == 'channels_first' else self.dimensions + 1
        self.group_axis = self.dimensions + 1 + (self.data_format == 'channels_first')

        # set during build
        self.group_valued_input = None
        self.repeated_bias_indices = None
        self.transformed_kernel_indices = None

    def call(self, inputs):
        inputs = self._merge_group_axis_and_pad(inputs)
        kernel = self._transformed_kernel()
        bias = self._repeated_bias()

        outputs = self._conv_call(inputs, kernel, bias)

        return self._restore_group_axis(outputs)

    def _merge_group_axis_and_pad(self, inputs):
        """
        If the input is a signal on the group, join the group axis with the channel axis.
        If necessary for equivariance, pad input.

        Shapes in 2D case (with default data_format='channels_last'):
        (batch, height, width, group.order, channels) -> (batch, height', width', group.order * channels)
        """
        if self.group_valued_input:
            inputs = self._merge_axes(inputs, merged_axis=self.group_axis, target_axis=self.channels_axis)
        inputs = self.equivariant_padding(inputs)
        return inputs

    def _transformed_kernel(self):
        """
        Perform the group action on the kernel, as a signal on the group, or on the grid in a lifting convolution.

        Shapes in 2D case:
        (height, width, group.order * channels_in, channels_out) ->
        (height, width, group.order * channels_in, subgroup.order * channels_out)
        """
        return tf.gather(tf.reshape(self.kernel, [-1]), indices=self.transformed_kernel_indices, axis=0)

    def _repeated_bias(self):
        """Transform the bias to subgroup.order repeated copies of itself."""
        return tf.gather(self.bias, indices=self.repeated_bias_indices, axis=0)

    def _restore_group_axis(self, outputs):
        """
        Reshape the output of the convolution, splitting off the group index from the channel axis.

        Shapes in 2D case (with default data_format='channels_last'):
        (batch, height, width, subgroup.order * channels) -> (batch, height, width, subgroup.order, channels)
        """
        group_channels_axis = self.channels_axis
        if self.group_valued_input and self.data_format == 'channels_last':
            group_channels_axis -= 1
        return self._split_axes(
            outputs, factor=self.subgroup.order, split_axis=group_channels_axis, target_axis=self.group_axis)

    def build(self, input_shape):
        self.group_valued_input = len(input_shape) == self.dimensions + 3  # this includes the batch dimension
        if self.data_format == 'channels_last' and self.group_valued_input:
            self.channels_axis += 1
        channels = input_shape[self.channels_axis]

        if self.group_valued_input:
            assert input_shape[self.group_axis] == self.group.order, \
                f'Got input shape {input_shape[self.group_axis]} in group axis {self.group_axis},' \
                f'expected {self.group.order}.'

            input_shape = self._merge_shapes(input_shape, merged_axis=self.group_axis, target_axis=self.channels_axis)

        super().build(input_shape)

        if self.group_valued_input:
            self.input_spec.axes = {self._get_channel_axis(): channels}

        self.equivariant_padding.build(input_shape)

        self.repeated_bias_indices = self._compute_repeated_bias_indices()
        self.transformed_kernel_indices = self._compute_transformed_kernel_indices()

    def _compute_repeated_bias_indices(self):
        """Compute a 1D tensor of indices used to gather from the bias in order to repeat it across the group axis."""
        indices = self.get_index_tensor(self.bias)
        indices = tf.concat([indices for _ in range(self.subgroup.order)], axis=0)
        return indices

    def _compute_transformed_kernel_indices(self):
        """Compute a tensor of indices used to gather from the kernel to produce the group action on it."""
        indices = self.get_index_tensor(self.kernel)

        axes_info = {'new_group_axis': self.dimensions,
                     'spatial_axes': tuple(d for d in range(self.dimensions))}

        if not self.group_valued_input:
            indices = self.group.action(indices, subgroup=self.subgroup_name, **axes_info)
        else:
            indices = self._restore_kernel_group_axis(indices)
            indices = self.group.action(indices, group_axis=self.dimensions, subgroup=self.subgroup_name, **axes_info)
            indices = self._merge_kernel_group_axis(indices)

        indices = self._merge_group_channels_out(indices)
        return indices

    def _restore_kernel_group_axis(self, kernel):
        """
        Shapes in 2D:
        (height, width, group.order * in_channels, out_channels) ->
        (height, width, group.order, in_channels, out_channels)
        """
        group_channel_axis = self.dimensions
        group_axis = self.dimensions
        return self._split_axes(kernel, factor=self.group.order, split_axis=group_channel_axis, target_axis=group_axis)

    def _merge_kernel_group_axis(self, kernel):
        """
        Shapes in 2D:
        (height, width, subgroup.order, group.order, in_channels, out_channels) ->
        (height, width, subgroup.order, group.order * in_channels, out_channels)
        """
        group_axis = self.dimensions + 1  # here and below extra +1 because the subgroup axis is inserted before
        channels_in_axis = self.dimensions + 1 + 1
        return self._merge_axes(kernel, merged_axis=group_axis, target_axis=channels_in_axis)

    def _merge_group_channels_out(self, kernel):
        """
        Shapes in 2D:
        (height, width, subgroup.order, group.order * in_channels, out_channels) ->
        (height, width, group.order * in_channels, subgroup.order * out_channels)
        """
        group_axis = self.dimensions
        channels_out_axis = self.dimensions + 2
        return self._merge_axes(kernel, merged_axis=group_axis, target_axis=channels_out_axis)

    @staticmethod
    def get_index_tensor(tensor):
        """
        Return a tensor of indices that reproduces the input through
        `tf.gather(tf.reshape(tensor, -1), indices=indices)`
        """
        return tf.reshape(tf.range(tf.size(tensor)), tensor.shape)

    def _merge_axes(self, tensor, merged_axis: int, target_axis: int):
        """Transpose the merge_axis to the left of the target_axis and then merge them."""
        shape = self._merge_shapes(tensor.shape, merged_axis=merged_axis, target_axis=target_axis)
        shape = [s if s is not None else -1 for s in shape]
        transposed_tensor = self._move_axis_to_left_of(tensor, moved_axis=merged_axis, target_axis=target_axis)
        return tf.reshape(transposed_tensor, shape)

    @staticmethod
    def _merge_shapes(shape, merged_axis: int, target_axis: int):
        shape = list(shape)
        shape[target_axis] *= shape[merged_axis]
        shape.pop(merged_axis)
        return shape

    def _split_axes(self, tensor, factor: int, split_axis: int, target_axis: int):
        """
        Split split_axis, dividing its size by factor, putting the new axis directly to its left,
        which is then moved to the target_axis index.
        """
        shape = list(tensor.shape)
        shape[split_axis] //= factor
        shape = tf.TensorShape(shape[:split_axis] + [factor] + shape[split_axis:])
        shape = [s if s is not None else -1 for s in shape]
        tensor_reshaped = tf.reshape(tensor, shape)
        tensor_transposed = self._move_axis_to_left_of(tensor_reshaped, moved_axis=split_axis, target_axis=target_axis)
        return tensor_transposed

    @staticmethod
    def _move_axis_to_left_of(tensor, moved_axis: int, target_axis: int):
        """Puts the moved_axis to the left of the target_axis, leaving the order of the other axes invariant."""
        axes = tuple(range(tensor.shape.rank))
        axes = axes[:moved_axis] + axes[moved_axis + 1:]
        if moved_axis < target_axis:
            target_axis -= 1
        axes = axes[:target_axis] + (moved_axis,) + axes[target_axis:]
        return tf.transpose(tensor, axes)

    def get_config(self):
        config = super().get_config()
        config['group'] = self.group.name
        config['subgroup'] = self.subgroup.name
        config['allow_non_equivariance'] = self.equivariant_padding.allow_non_equivariance
        return config

    def _conv_call(self, reshaped_inputs, reshaped_kernel, reshaped_bias):
        """
        This is the original call method, with replacements:
        - self.kernel -> reshaped_kernel (argument)
        - self.bias -> reshaped_bias (argument)
        - self.filters -> filters_multiplied = self.filters * self.group.order
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
        out_shape = batch, height, width, channels * self.subgroup.order
        return tf.TensorShape(out_shape)


P4MConv2D = partial(GroupConv2D, group=wallpaper_groups.P4M)
P4Conv2D = partial(GroupConv2D, group=wallpaper_groups.P4)
P2MMConv2D = partial(GroupConv2D, group=wallpaper_groups.P2MM)
PMhConv2D = partial(GroupConv2D, group=wallpaper_groups.PMh)
PMwConv2D = partial(GroupConv2D, group=wallpaper_groups.PMw)
P2Conv2D = partial(GroupConv2D, group=wallpaper_groups.P2)
P1Conv2D = partial(GroupConv2D, group=wallpaper_groups.P1)
