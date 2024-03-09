from functools import partial

from tensorflow.keras.layers import Conv2D, Conv2DTranspose

from groco.groups import wallpaper_groups
from groco.layers import GroupTransforms
from groco.utils import backup_and_restore


class GroupConv2D(Conv2D):
    """
    Group convolution layer built on Keras's Conv2D layer.
    Additional arguments:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - subgroup: name of subgroup to act with (analog of a stride in the group direction), defaults to None,
                meaning full group is acted with
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    Expected input shape: (batch, height, width, group.order, in_channels),
    with the group.order axis omitted for the first layer (the lifting convolution).
    """

    def __init__(
        self, group, kernel_size, allow_non_equivariance: bool = False, subgroup="", **kwargs
    ):
        self.group_transforms = GroupTransforms(
            allow_non_equivariance=allow_non_equivariance,
            kernel_size=kernel_size,
            dimensions=2,
            group=group,
            subgroup=subgroup,
            **kwargs
        )
        kwargs["padding"] = self.group_transforms.built_in_padding_option
        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup

        super().__init__(kernel_size=kernel_size, **kwargs)
        self.group_valued_input = None

    @backup_and_restore(("kernel", "bias", "filters"))
    def call(self, inputs):
        inputs = self.group_transforms.merge_group_axis_and_pad(inputs)
        self.kernel = self.group_transforms.transform_kernel(self.kernel)
        if self.use_bias:
            self.bias = self.group_transforms.repeat_bias(self.bias)
        self.filters *= self.subgroup.order

        outputs = super().call(inputs)

        return self.group_transforms.restore_group_axis(outputs)

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.group_valued_input = self.group_transforms.group_valued_input
        super().build(reshaped_input)
        self.group_transforms.compute_conv_indices(
            input_shape, self.kernel, self.bias, self.use_bias
        )
        if self.group_valued_input:
            self.input_spec.axes = {
                self._get_channel_axis(): input_shape[self.group_transforms.channels_axis]
            }

    def get_config(self):
        config = super().get_config()
        config.update(self.group_transforms.get_config())
        return config


P4MConv2D = partial(GroupConv2D, group=wallpaper_groups.P4M)
P4Conv2D = partial(GroupConv2D, group=wallpaper_groups.P4)
P2MMConv2D = partial(GroupConv2D, group=wallpaper_groups.P2MM)
PMhConv2D = partial(GroupConv2D, group=wallpaper_groups.PMh)
PMwConv2D = partial(GroupConv2D, group=wallpaper_groups.PMw)
P2Conv2D = partial(GroupConv2D, group=wallpaper_groups.P2)
P1Conv2D = partial(GroupConv2D, group=wallpaper_groups.P1)


class GroupConv2DTranspose(Conv2DTranspose):
    """
    Group transpose convolution layer built on Keras's Conv2DTranspose layer.
    Additional arguments:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - subgroup: name of subgroup to act with (analog of a stride in the group direction), defaults to None,
                meaning full group is acted with
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    Expected input shape: (batch, height, width, group.order, in_channels),
    with the group.order axis omitted for the first layer (the lifting convolution).
    """

    def __init__(
        self, group, kernel_size, allow_non_equivariance: bool = False, subgroup="", **kwargs
    ):
        self.group_transforms = GroupTransforms(
            allow_non_equivariance=allow_non_equivariance,
            kernel_size=kernel_size,
            dimensions=2,
            group=group,
            subgroup=subgroup,
            transpose=True,
            **kwargs
        )
        kwargs["padding"] = self.group_transforms.built_in_padding_option
        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup

        super().__init__(kernel_size=kernel_size, **kwargs)
        self.group_valued_input = None

    @backup_and_restore(("kernel", "bias", "filters"))
    def call(self, inputs):
        inputs = self.group_transforms.merge_group_axis_and_pad(inputs)
        self.kernel = self.group_transforms.transform_kernel(self.kernel)
        if self.use_bias:
            self.bias = self.group_transforms.repeat_bias(self.bias)
        self.filters *= self.group.order

        outputs = super().call(inputs)

        return self.group_transforms.restore_group_axis(outputs)

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.group_valued_input = self.group_transforms.group_valued_input
        super().build(reshaped_input)
        self.group_transforms.compute_conv_indices(
            input_shape, self.kernel, self.bias, self.use_bias
        )
        if self.group_valued_input:
            self.input_spec.axes = {
                self._get_channel_axis(): input_shape[self.group_transforms.channels_axis]
            }
            self.input_spec.ndim += 1

    def get_config(self):
        config = super().get_config()
        config.update(self.group_transforms.get_config())
        return config
