from functools import partial

from keras.layers import Conv3D

from groco.groups import space_groups
from groco.layers.group_transforms import GroupTransforms
from groco.utils import backup_and_restore


class GroupConv3D(Conv3D):
    """
    Group convolution layer built on Keras's Conv3D layer.

    Args:
        group: one of the space groups, string name or the object itself.
        subgroup: name of subgroup to act with (analog of a stride in the group direction).
            Defaults to None, meaning full group is acted with.
        padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal
            extra amount to maintain equivariance.
        allow_non_equivariance: set to true along with setting padding to either `valid` or `same`
            to insist using non-equivariant padding.
        all Conv3D args.

    Input shape:
    - If `data_format="channels_last"`:
        A 6D (5D) tensor with shape: `(batch_size, height, width, group.order, channels)`
        with the group.order axis omitted for the first layer (the lifting convolution).
    - If `data_format="channels_first"`:
        A 6D (5D) tensor with shape: `(batch_size, channels, height, width, group.order)`
        with the group.order axis omitted for the first layer (the lifting convolution).

    Output shape:
    - If `data_format="channels_last"`:
        A 6D (5D) tensor with shape: `(batch_size, new_height, new_width, subgroup.order, filters)`
    - If `data_format="channels_first"`:
        A 6D (5D) tensor with shape: `(batch_size, filters, new_height, new_width, subgroup.order)`

    Returns:
        A 6D tensor representing `activation(conv2d(inputs, kernel) + bias)`.
    """

    def __init__(
        self, group, kernel_size, allow_non_equivariance: bool = False, subgroup="", **kwargs
    ):
        dimensions = 3
        transpose = False
        self.group_transforms = GroupTransforms(
            allow_non_equivariance=allow_non_equivariance,
            kernel_size=kernel_size,
            dimensions=dimensions,
            group=group,
            subgroup=subgroup,
            transpose=transpose,
            **kwargs,
        )
        kwargs["padding"] = self.group_transforms.built_in_padding_option
        super().__init__(kernel_size=kernel_size, **kwargs)

        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup

        self.group_valued_input = None

    @backup_and_restore(("kernel", "bias", "filters"))
    def call(self, inputs):
        self.kernel, self.bias, self.filters, inputs = self.group_transforms.prepare_call(
            self.kernel, self.bias, self.filters, inputs, self.use_bias
        )
        outputs = super().call(inputs)
        return self.group_transforms.restore_group_axis(outputs)

    def build(self, input_shape):
        self.group_transforms.build(input_shape)
        self.group_valued_input = self.group_transforms.group_valued_input
        super().build(self.group_transforms.reshaped_input)
        self.group_transforms.compute_conv_indices(
            input_shape, self.kernel, self.bias, self.use_bias
        )
        if self.group_valued_input:
            if self.data_format == "channels_first":
                channel_axis = -1 - self.rank
            else:
                channel_axis = -1
            self.input_spec.axes = {channel_axis: input_shape[self.group_transforms.channels_axis]}

    def get_config(self):
        config = super().get_config()
        config.update(self.group_transforms.get_config())
        return config

    def compute_output_shape(self, input_shape):
        input_shape_merged = self.group_transforms.reshaped_input
        output_shape = super().compute_output_shape(input_shape_merged)
        return self.group_transforms.correct_output_shape(output_shape)


OhConv3D = partial(GroupConv3D, group=space_groups.Oh)
OConv3D = partial(GroupConv3D, group=space_groups.O)
D4hConv3D = partial(GroupConv3D, group=space_groups.D4h)
D4Conv3D = partial(GroupConv3D, group=space_groups.D4)
