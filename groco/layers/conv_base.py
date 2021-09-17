import tensorflow as tf
from groco.layers import EquivariantPadding
from tensorflow.keras.layers import Layer
from groco.groups import Group, group_dict


class GroupTransforms(Layer):
    """
    Helper layer meant only for use within other layers involving group operations.
    Takes care of all group related transformations.

    All public methods involve at most a single tf.gather and tf.reshape call, using precomputed indices.

    Methods:
        merge_group_axis_and_pad
        repeat_bias
        transform_kernel
        restore_group_axis
        subgroup_pooling
    Methods used during build:
        build
        compute_conv_indices
        compute_pooling_indices
    """

    def __init__(self, group, kernel_size, dimensions: int, data_format='channels_last',
                 allow_non_equivariance: bool = False, subgroup=None, transpose=False, **kwargs):
        self.group = group if isinstance(group, Group) else group_dict[group]
        self.subgroup = self.group if subgroup is None else group_dict[subgroup]
        self.dimensions = dimensions

        self.equivariant_padding = EquivariantPadding(
            allow_non_equivariance=allow_non_equivariance, kernel_size=kernel_size, dimensions=dimensions,
            transpose=transpose, **kwargs)
        self.built_in_padding_option = self.equivariant_padding.built_in_padding_option
        self.transpose = transpose

        super().__init__()
        self.data_format = data_format
        # axes refer to input
        self.channels_axis = 1 if self.data_format == 'channels_first' else self.dimensions + 1
        self.group_axis = self.dimensions + 1 + (self.data_format == 'channels_first')

        # set during build
        self.group_valued_input = None
        self.conv_input_shape = None
        self._repeated_bias_indices = None
        self._transformed_kernel_indices = None
        self._input_indices = None
        self._output_indices = None
        self._pooling_indices = None

    def merge_group_axis_and_pad(self, inputs):
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

    def repeat_bias(self, bias):
        """Transform the bias to subgroup.order repeated copies of itself."""
        return tf.gather(bias, indices=self._repeated_bias_indices, axis=0)

    def transform_kernel(self, kernel):
        """
        Perform the group action on the kernel, as a signal on the group, or on the grid in a lifting convolution.

        Shapes in 2D case:
        (height, width, group.order * channels_in, channels_out) ->
        (height, width, group.order * channels_in, subgroup.order * channels_out)
        """
        return tf.gather(tf.reshape(kernel, [-1]), indices=self._transformed_kernel_indices, axis=0)

    def restore_group_axis(self, outputs, subgroup=True):
        """
        Reshape the output of the convolution, splitting off the group index from the channel axis.

        Shapes in 2D case (with default data_format='channels_last'):
        (batch, height, width, subgroup.order * channels) -> (batch, height, width, subgroup.order, channels)
        """
        group_channels_axis = self.channels_axis
        if self.group_valued_input and self.data_format == 'channels_last':
            group_channels_axis -= 1
        group_axis = self.group_axis + (self.data_format == 'channels_first')
        order = self.subgroup.order if subgroup else self.group.order
        return self._split_axes(outputs, factor=order, split_axis=group_channels_axis, target_axis=group_axis)

    def subgroup_pooling(self, inputs, pool_type: str):
        """
        Pool in the group direction, over a set of coset representatives, keeping the subgroup.

        Shapes in 2D case (with default data_format='channels_last'):
        (batch, height, width, group.order, channels) -> (batch, height, width, subgroup.order, channels)
        """
        outputs = tf.gather(inputs, axis=self.group_axis, indices=self._pooling_indices)
        pooling = tf.reduce_max if pool_type == 'max' else tf.reduce_mean
        outputs = pooling(outputs, axis=self.group_axis + 1)
        return outputs

    def build(self, input_shape):
        self.group_valued_input = len(input_shape) == self.dimensions + 3  # this includes the batch dimension
        if self.data_format == 'channels_last' and self.group_valued_input:
            self.channels_axis += 1

        if self.group_valued_input:
            assert input_shape[self.group_axis] == self.group.order, \
                f'Got input shape {input_shape[self.group_axis]} in group axis {self.group_axis},' \
                f'expected {self.group.order}.'

            reshaped_input = self._merge_shapes(
                input_shape, merged_axis=self.group_axis, target_axis=self.channels_axis)
        else:
            reshaped_input = input_shape
        return reshaped_input

    def compute_conv_indices(self, input_shape, kernel, bias):
        self._repeated_bias_indices = self._compute_repeated_bias_indices(bias)
        self._transformed_kernel_indices = self._compute_transformed_kernel_indices(kernel)

    def compute_pooling_indices(self):
        indices = tf.gather(self.group.composition, axis=1, indices=self.group.cosets[self.subgroup.name])
        subgroup_indices = self.group.subgroup[self.subgroup.name]
        self._pooling_indices = tf.gather(indices, axis=0, indices=subgroup_indices)

    def _compute_repeated_bias_indices(self, bias):
        """Compute a 1D tensor of indices used to gather from the bias in order to repeat it across the group axis."""
        indices = self._get_index_tensor(bias)
        indices = tf.concat([indices for _ in range(self.subgroup.order)], axis=0)
        return indices

    def _compute_transformed_kernel_indices(self, kernel):
        """Compute a tensor of indices used to gather from the kernel to produce the group action on it."""
        if self.transpose:
            kernel = self._switch_in_out(kernel)
        indices = self._get_index_tensor(kernel)

        axes_info = {'new_group_axis': self.dimensions,
                     'spatial_axes': tuple(d for d in range(self.dimensions))}

        if not self.group_valued_input:
            indices = self.group.action(indices, subgroup=self.subgroup.name, **axes_info)
        else:
            indices = self._restore_kernel_group_axis(indices)
            indices = self.group.action(indices, group_axis=self.dimensions, subgroup=self.subgroup.name, **axes_info)
            indices = self._merge_kernel_group_axis(indices)

        indices = self._merge_group_channels_out(indices)
        if self.transpose:
            indices = self._switch_in_out(indices)
        return indices

    @staticmethod
    def _switch_in_out(kernel):
        axes = list(range(kernel.shape.rank))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return tf.transpose(kernel, tuple(axes))

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
    def _get_index_tensor(tensor):
        """
        Return a tensor of indices that reproduces the input through
        `tf.gather(tf.reshape(tensor, -1), indices=indices)`
        """
        if isinstance(tensor, tf.TensorShape):
            return tf.reshape(tf.range(tf.reduce_prod(tensor)), tensor)
        else:
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
        config = {'group': self.group.name, 'subgroup': self.subgroup.name}
        config.update(self.equivariant_padding.get_config())
        return config
