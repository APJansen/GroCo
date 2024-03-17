import keras
from keras import ops

from groco.groups import space_group_dict, wallpaper_group_dict
from groco.layers import GroupConv2D, GroupConv2DTranspose, GroupConv3D, GroupConv3DTranspose
from groco.utils import check_equivariance
from tests.custom_testcase import KerasTestCase as TestCase


def get_conv_layer(dimension: int, transpose: bool):
    if dimension == 2:
        return GroupConv2DTranspose if transpose else GroupConv2D
    elif dimension == 3:
        return GroupConv3DTranspose if transpose else GroupConv3D


class TestGroupConv3D(TestCase):
    def __init__(self, tests):
        super().__init__(tests)
        self.dimension = 3
        self.transpose = False
        self.conv = get_conv_layer(self.dimension, self.transpose)

        self.spatial_axes = tuple(range(1, self.dimension + 1))
        self.group_axis = self.dimension + 1

        if self.dimension == 2:
            self.group_dict = wallpaper_group_dict
            self.example_group = self.group_dict["P4"]
        elif self.dimension == 3:
            self.group_dict = space_group_dict
            self.example_group = self.group_dict["D4"]

        self.filters = 4
        self.xsize = 3
        self.input_features = 2
        self.shape = (1,) + (self.xsize,) * self.dimension + (self.input_features,)

    def test_lift_shape(self):
        signal_on_grid = keras.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            conv_layer = self.conv(
                group=group, kernel_size=3, filters=self.filters, padding="same_equiv"
            )
            signal_on_group = conv_layer(signal_on_grid)
            self.assertEqual(signal_on_group.shape, self.shape[:-1] + (group.order, self.filters))

    def test_lift_shape_subgroup(self):
        signal_on_grid = keras.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
                conv_layer = self.conv(
                    group=group,
                    kernel_size=3,
                    filters=self.filters,
                    padding="same_equiv",
                    subgroup=subgroup_name,
                )

                signal_on_group = conv_layer(signal_on_grid)
                out_size = group.order if self.transpose else subgroup.order
                self.assertEqual(signal_on_group.shape, self.shape[:-1] + (out_size, self.filters))

    def test_gc_shape(self):
        for group in self.group_dict.values():
            signal_on_group = keras.random.normal(
                shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
            )
            conv_layer = self.conv(
                group=group, kernel_size=3, filters=self.filters, padding="same_equiv"
            )
            new_signal = conv_layer(signal_on_group)
            self.assertEqual(new_signal.shape, signal_on_group.shape[:-1] + (self.filters,))

    def test_gc_shape_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                out_size = self.group_dict[subgroup_name].order if self.transpose else group.order
                signal = keras.random.normal(
                    shape=self.shape[:-1] + (out_size, self.shape[-1]), seed=42
                )
                conv_layer = self.conv(
                    group=group,
                    kernel_size=3,
                    filters=self.filters,
                    padding="same_equiv",
                    subgroup=subgroup_name,
                )

                new_signal = conv_layer(signal)
                out_size = group.order if self.transpose else self.group_dict[subgroup_name].order
                self.assertEqual(new_signal.shape, self.shape[:-1] + (out_size, self.filters))

    def test_lift_equiv(self):
        signal_on_grid = keras.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            conv_layer = self.conv(
                group=group, kernel_size=3, filters=self.filters, padding="same_equiv"
            )
            conv_layer(signal_on_grid)
            conv_layer.bias = 1 + keras.random.normal(shape=conv_layer.bias.shape)
            equiv_diff = check_equivariance(
                conv_layer,
                signal_on_grid,
                spatial_axes=self.spatial_axes,
                group_axis=self.group_axis,
                domain_group=None,
            )
            self.assertAllLess(equiv_diff, 1e-4)

    def test_lift_equiv_subgroup(self):
        signal_on_grid = keras.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                conv_layer = self.conv(
                    group=group,
                    kernel_size=3,
                    filters=self.filters,
                    padding="same_equiv",
                    subgroup=subgroup_name,
                )
                conv_layer(signal_on_grid)
                conv_layer.bias = 1 + keras.random.normal(shape=conv_layer.bias.shape)
                equiv_diff = check_equivariance(
                    conv_layer,
                    signal_on_grid,
                    spatial_axes=self.spatial_axes,
                    group_axis=self.group_axis,
                    acting_group=subgroup_name,
                    target_group=group.name if self.transpose else subgroup_name,
                    domain_group=None,
                )
                self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv(self):
        for group in self.group_dict.values():
            signal_on_group = keras.random.normal(
                shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
            )
            conv_layer = self.conv(
                group=group, kernel_size=3, filters=self.filters, padding="same_equiv"
            )
            equiv_diff = check_equivariance(
                conv_layer,
                signal_on_group,
                group_axis=self.group_axis,
                spatial_axes=self.spatial_axes,
            )
            self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                out_size = self.group_dict[subgroup_name].order if self.transpose else group.order
                signal = keras.random.normal(
                    shape=self.shape[:-1] + (out_size, self.shape[-1]), seed=42
                )
                conv_layer = self.conv(
                    group=group,
                    kernel_size=3,
                    filters=self.filters,
                    padding="same_equiv",
                    subgroup=subgroup_name,
                )
                equiv_diff = check_equivariance(
                    conv_layer,
                    signal,
                    group_axis=self.group_axis,
                    spatial_axes=self.spatial_axes,
                    acting_group=subgroup_name,
                    target_group=group.name if self.transpose else subgroup_name,
                    domain_group=subgroup_name if self.transpose else group.name,
                )

                self.assertAllLess(equiv_diff, 1e-4)

    def test_padding_equiv(self):
        for padding in ["same_equiv", "valid_equiv"]:
            # TODO: figure out why it's not working for all strides for transpose convolutions
            stride_list = [3] if self.transpose else [3, 5, 7]
            for strides in stride_list:
                group = self.example_group
                signal_on_group = keras.random.normal(
                    shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
                )
                conv_layer = self.conv(
                    group=group,
                    kernel_size=3,
                    strides=strides,
                    filters=self.filters,
                    padding=padding,
                )
                equiv_diff = check_equivariance(
                    conv_layer,
                    signal_on_group,
                    group_axis=self.group_axis,
                    spatial_axes=self.spatial_axes,
                )
                self.assertAllLess(equiv_diff, 1e-4)
