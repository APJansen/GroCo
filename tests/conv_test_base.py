import keras
from keras import ops

from groco.groups import space_group_dict, wallpaper_group_dict
from groco.layers import GroupConv2D, GroupConv2DTranspose, GroupConv3D, GroupConv3DTranspose
from groco.utils import check_equivariance
from tests.custom_testcase import KerasTestCase as TestCase


def get_attributes(dimension: int, transpose: bool):
    if dimension == 2:
        conv_layer = GroupConv2DTranspose if transpose else GroupConv2D
        group_dict = wallpaper_group_dict
        example_group = group_dict["P4"]
    elif dimension == 3:
        conv_layer = GroupConv3DTranspose if transpose else GroupConv3D
        group_dict = space_group_dict
        example_group = group_dict["D4"]

    return conv_layer, group_dict, example_group


class TestConvBase:
    dimension = None
    transpose = None

    def __init__(self, tests):
        super().__init__(tests)

        self.conv, self.group_dict, self.example_group = get_attributes(
            self.dimension, self.transpose
        )

        self.spatial_axes = tuple(range(1, self.dimension + 1))
        self.group_axis = self.dimension + 1

        self.filters = 4
        self.kernel_size = 3
        self.xsize = 3
        self.input_features = 2
        self.shape = (1,) + (self.xsize,) * self.dimension + (self.input_features,)

    def test_lift_shape(self):
        signal_on_grid = self.generate_signal(None)
        for group in self.group_dict.values():
            conv_layer = self.generate_layer(group)
            signal_on_group = conv_layer(signal_on_grid)
            self.assertEqual(signal_on_group.shape, self.shape[:-1] + (group.order, self.filters))

    def test_lift_shape_subgroup(self):
        signal_on_grid = self.generate_signal(None)
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
                conv_layer = self.generate_layer(group, subgroup=subgroup_name)

                signal_on_group = conv_layer(signal_on_grid)
                out_size = group.order if self.transpose else subgroup.order
                self.assertEqual(signal_on_group.shape, self.shape[:-1] + (out_size, self.filters))

    def test_gc_shape(self):
        for group in self.group_dict.values():
            signal_on_group = self.generate_signal(group)
            conv_layer = self.generate_layer(group)
            new_signal = conv_layer(signal_on_group)
            self.assertEqual(new_signal.shape, signal_on_group.shape[:-1] + (self.filters,))

    def test_gc_shape_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                out_group = self.group_dict[subgroup_name] if self.transpose else group
                signal = self.generate_signal(out_group)
                conv_layer = self.generate_layer(group, subgroup=subgroup_name)

                new_signal = conv_layer(signal)
                out_size = group.order if self.transpose else self.group_dict[subgroup_name].order
                self.assertEqual(new_signal.shape, self.shape[:-1] + (out_size, self.filters))

    def test_lift_equiv(self):
        signal_on_grid = self.generate_signal(None)
        for group in self.group_dict.values():
            conv_layer = self.conv(
                group=group, kernel_size=3, filters=self.filters, padding="same_equiv"
            )
        self.check_equivariance(conv_layer, signal_on_grid, domain_group=None)

    def test_lift_equiv_subgroup(self):
        signal_on_grid = self.generate_signal(None)
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                conv_layer = self.generate_layer(group, subgroup=subgroup_name)
                self.check_equivariance(
                    conv_layer,
                    signal_on_grid,
                    domain_group=None,
                    target_group=group.name if self.transpose else subgroup_name,
                    acting_group=subgroup_name,
                )

    def test_gc_equiv(self):
        for group in self.group_dict.values():
            signal_on_group = self.generate_signal(group)
            conv_layer = self.generate_layer(group)
            self.check_equivariance(conv_layer, signal_on_group)

    def test_gc_equiv_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                out_group = self.group_dict[subgroup_name] if self.transpose else group
                signal = self.generate_signal(out_group)
                conv_layer = self.generate_layer(group, subgroup=subgroup_name)
                self.check_equivariance(
                    conv_layer,
                    signal,
                    acting_group=subgroup_name,
                    target_group=group.name if self.transpose else subgroup_name,
                    domain_group=subgroup_name if self.transpose else group.name,
                )

    def test_padding_equiv(self):
        for padding in ["same_equiv", "valid_equiv"]:
            # TODO: figure out why it's not working for all strides for transpose convolutions
            stride_list = [3] if self.transpose else [3, 5, 7]
            for strides in stride_list:
                group = self.example_group
                signal_on_group = self.generate_signal(group)
                conv_layer = self.generate_layer(group, padding=padding, strides=strides)
                self.check_equivariance(conv_layer, signal_on_group)

    def generate_signal(self, group):
        if type(group) == str:
            group = self.group_dict[group]
        if group == None:
            shape = self.shape
        else:
            shape = self.shape[:-1] + (group.order, self.shape[-1])
        return keras.random.normal(shape=shape, seed=42)

    def generate_layer(self, group, padding="same_equiv", strides=1, subgroup=""):
        return self.conv(
            group=group,
            kernel_size=self.kernel_size,
            filters=self.filters,
            padding=padding,
            strides=strides,
            subgroup=subgroup,
        )

    def check_equivariance(self, layer, signal, domain_group="", acting_group="", target_group=""):
        layer(signal)  # building
        layer.bias = 1 + keras.random.normal(shape=layer.bias.shape)  # add random bias
        equiv_diff = check_equivariance(
            layer,
            signal,
            spatial_axes=self.spatial_axes,
            group_axis=self.group_axis,
            domain_group=domain_group,
            acting_group=acting_group,
            target_group=target_group,
        )
        self.assertAllLess(equiv_diff, 1e-4)
