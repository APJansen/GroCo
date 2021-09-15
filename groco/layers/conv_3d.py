from tensorflow.keras.layers import Conv3D
from functools import partial
from groco.layers import GroupTransforms, conv_call
from groco.groups import space_groups, group_dict


class GroupConv3D(Conv3D):
    """
    Group convolution layer built on Keras's Conv3D layer.
    Additional arguments:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - subgroup: name of subgroup to act with (analog of a stride in the group direction), defaults to None,
                meaning full group is acted with
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    Expected input shape: (batch, height, width, depth, group.order, in_channels),
    with the group.order axis omitted for the first layer (the lifting convolution).
    """
    def __init__(self, group, kernel_size, allow_non_equivariance: bool = False, subgroup=None, **kwargs):
        self.group_transforms = GroupTransforms(
            allow_non_equivariance=allow_non_equivariance, kernel_size=kernel_size, dimensions=3,
            group=group, subgroup=subgroup, **kwargs)
        kwargs['padding'] = self.group_transforms.built_in_padding_option
        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup

        super().__init__(kernel_size=kernel_size, **kwargs)
        self.group_valued_input = None

    def call(self, inputs):
        inputs = self.group_transforms.merge_group_axis_and_pad(inputs)
        kernel = self.group_transforms.transform_kernel(self.kernel)
        bias = self.group_transforms.repeat_bias(self.bias)

        outputs = conv_call(self, inputs, kernel, bias)  # to avoid duplication this is not a class method

        return self.group_transforms.restore_group_axis(outputs)

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.group_valued_input = self.group_transforms.group_valued_input
        super().build(reshaped_input)
        self.group_transforms.compute_conv_indices(input_shape, self.kernel, self.bias)
        if self.group_valued_input:
            self.input_spec.axes = {self._get_channel_axis(): input_shape[self.group_transforms.channels_axis]}

    def get_config(self):
        config = super().get_config()
        config.update(self.group_transforms.get_config())
        return config

    def _compute_output_shape(self, input_shape):
        """Called in conv_call to set an intermediate output shape."""
        return self.group_transforms.multiply_channels(super().compute_output_shape(input_shape))


OhConv3D = partial(GroupConv3D, group=space_groups.Oh)
OConv3D = partial(GroupConv3D, group=space_groups.O)
D4hConv3D = partial(GroupConv3D, group=space_groups.D4h)
D4Conv3D = partial(GroupConv3D, group=space_groups.D4)
