from keras import ops
from keras.layers import Layer

from groco import utils
from groco.groups import Group, group_dict
from groco.layers.padding import EquivariantPadding


class GroupTransforms:
    """
    Helper layer meant only for use within other layers involving group operations.
    Takes care of all group related transformations.

    All public methods involve at most a single ops.take and ops.reshape call, using precomputed indices.

    Methods:
        merge_group_axis_and_pad
        repeat_bias
        transform_kernel
        restore_group_axis
        subgroup_pooling
    Methods used during build:
        build
        build_conv
        build_pool
    """

    def __init__(
        self,
        group,
        kernel_size,
        dimensions: int,
        data_format="channels_last",
        allow_non_equivariance: bool = False,
        subgroup="",
        transpose=False,
        separable=False,
        **kwargs,
    ):
        super().__init__()

        self.dimensions = dimensions
        self.transpose = transpose
        self.separable = separable

        self.group = group if isinstance(group, Group) else group_dict[group]
        self.subgroup = group_dict[self.group.parse_subgroup(subgroup)]
        self.domain_group, self.acting_group = self.group.name, self.subgroup.name
        if transpose:
            self.domain_group, self.acting_group = self.acting_group, self.domain_group

        self.equivariant_padding = EquivariantPadding(
            allow_non_equivariance=allow_non_equivariance,
            kernel_size=kernel_size,
            dimensions=dimensions,
            transpose=transpose,
            **kwargs,
        )
        self.built_in_padding_option = self.equivariant_padding.built_in_padding_option

        self.data_format = data_format
        # axes refer to input
        self.channels_axis = 1 if self.data_format == "channels_first" else self.dimensions + 1
        self.group_axis = self.dimensions + 1 + (self.data_format == "channels_first")

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
        (batch, height, width, domain_group.order, channels) ->
        (batch, height', width', domain_group.order * channels)
        """
        if self.group_valued_input:
            inputs = utils.merge_axes(
                inputs, merged_axis=self.group_axis, target_axis=self.channels_axis
            )
        inputs = self.equivariant_padding(inputs)
        return inputs

    def repeat_bias(self, bias):
        """Transform the bias to subgroup.order repeated copies of itself."""
        return ops.take(bias, indices=self._repeated_bias_indices, axis=0)

    def transform_kernel(self, kernel):
        """
        Perform the group action on the kernel, as a signal on the group, or on the grid in a lifting convolution.

        Shapes in 2D case:
        (height, width, domain_group.order * channels_in, channels_out) ->
        (height, width, domain_group.order * channels_in, acting_group.order * channels_out)
        """
        return ops.take(ops.reshape(kernel, [-1]), indices=self._transformed_kernel_indices, axis=0)

    def restore_group_axis(self, outputs):
        """
        Reshape the output of the convolution, splitting off the group index from the channel axis.

        Shapes in 2D case (with default data_format='channels_last'):
        (batch, height, width, acting_group.order * channels) ->
        (batch, height, width, acting_group.order, channels)
        """
        group_channels_axis = self.channels_axis
        if self.group_valued_input and self.data_format == "channels_last":
            group_channels_axis -= 1
        factor = len(self.group.subgroup[self.acting_group])
        out = utils.split_axes(outputs, left_size=factor, right_axis=group_channels_axis)
        out = ops.moveaxis(out, group_channels_axis, self.group_axis)
        return out

    def subgroup_pooling(self, inputs, pool_type: str):
        """
        Pool in the group direction, over a set of coset representatives, keeping the subgroup.

        Shapes in 2D case (with default data_format='channels_last'):
        (batch, height, width, group.order, channels) -> (batch, height, width, subgroup.order, channels)
        """
        outputs = ops.take(inputs, axis=self.group_axis, indices=self._pooling_indices)
        pooling = ops.max if pool_type == "max" else ops.mean
        outputs = pooling(outputs, axis=self.group_axis + 1)
        return outputs

    def build(self, input_shape):
        # apart from the spatial dimensions, there are feature and batch axes
        # if there is a third axis, it is the group axis and the input is a signal on the group
        self.group_valued_input = len(input_shape) == self.dimensions + 3
        if not self.group_valued_input:
            self.domain_group = None
        if self.data_format == "channels_last" and self.group_valued_input:
            self.channels_axis += 1

        if self.group_valued_input:
            order = len(self.group.subgroup[self.domain_group])
            assert input_shape[self.group_axis] == order, (
                f"Got input shape {input_shape[self.group_axis]} in group axis {self.group_axis},"
                f"expected {order}."
            )

            reshaped_input = utils.merge_shapes(
                input_shape, merged_axis=self.group_axis, target_axis=self.channels_axis
            )
        else:
            reshaped_input = input_shape
        self.reshaped_input = reshaped_input
        self.equivariant_padding.build(reshaped_input)

    def build_conv(self, kernel, bias, use_bias):
        if use_bias:
            self._repeated_bias_indices = self._compute_repeated_bias_indices(bias)
        self._transformed_kernel_indices = self._compute_transformed_kernel_indices(kernel)

    def build_pool(self):
        indices = ops.take(
            self.group.composition, axis=1, indices=self.group.cosets[self.subgroup.name]
        )
        subgroup_indices = self.group.subgroup[self.subgroup.name]
        self._pooling_indices = ops.take(indices, axis=0, indices=subgroup_indices)

    def _compute_repeated_bias_indices(self, bias):
        """Compute a 1D tensor of indices used to gather from the bias in order to repeat it across the group axis."""
        indices = utils.get_index_tensor(bias)
        order = len(self.group.subgroup[self.acting_group])
        indices = ops.concatenate([indices for _ in range(order)], axis=0)
        return indices

    def _compute_transformed_kernel_indices(self, kernel):
        """
        Compute a tensor of indices used to gather from the kernel to produce the group action on it.
        """
        if self.transpose:
            kernel = ops.swapaxes(kernel, -1, -2)

        indices = utils.get_index_tensor(kernel)

        kwargs = {
            "new_group_axis": self.dimensions,
            "spatial_axes": tuple(d for d in range(self.dimensions)),
            "domain_group": self.domain_group,
            "acting_group": self.acting_group,
        }

        if not self.group_valued_input:
            indices = self.group.action(indices, **kwargs)
        else:
            # Split the group axis from the channel axis, resulting in shape:
            # (*spatial, domain_group.order, in_channels, out_channels)
            factor = len(self.group.subgroup[self.domain_group])
            indices = utils.split_axes(indices, left_size=factor, right_axis=-2)

            # Act on the group axis, resulting in shape:
            # (*spatial, acting_group.order, domain_group.order, in_channels, out_channels)
            indices = self.group.action(indices, group_axis=self.dimensions, **kwargs)

            # Merge back the domain group axis with the in_channel axis, resulting in shape:
            # (*spatial, acting_group.order, domain_group.order * in_channels, out_channels)
            indices = utils.merge_axes(indices, merged_axis=-3, target_axis=-2)

        # Merge the acting group axis with the out_channel axis, resulting in shape:
        # (*spatial, domain_group.order * in_channels, acting_group.order * out_channels)
        indices = utils.merge_axes(indices, merged_axis=-3, target_axis=-1)

        if self.transpose:
            indices = ops.swapaxes(indices, -1, -2)

        return indices

    def correct_output_shape(self, output_shape):
        """Insert the group axis in the correct place."""
        output_shape = list(output_shape)
        return (
            output_shape[: self.group_axis] + [self.group.order] + output_shape[self.group_axis :]
        )

    def get_config(self):
        config = {"group": self.group, "subgroup": self.subgroup.name}
        config.update(self.equivariant_padding.get_config())
        return config

    def prepare_call(self, kernel, bias, filters, inputs, use_bias):
        inputs = self.merge_group_axis_and_pad(inputs)
        kernel = self.transform_kernel(kernel)
        if use_bias:
            bias = self.repeat_bias(bias)
        factor = len(self.group.subgroup[self.acting_group])
        filters *= factor
        return kernel, bias, filters, inputs
