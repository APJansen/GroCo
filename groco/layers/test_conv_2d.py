from tensorflow.test import TestCase
from groco.groups import wallpaper_group_dict
from groco.layers import GroupConv2D, GroupConv2DTranspose
from groco.utils import test_equivariance
import tensorflow as tf


class TestGroupConv2D(TestCase):
    def __init__(self, tests):
        super().__init__(tests)
        self.shape = (11, 7, 7, 3)
        self.spatial_axes = (1, 2)
        self.group_axis = 3  # for signals on group, not present in shape above
        self.filters = 5
        self.conv = GroupConv2D
        self.group_dict = wallpaper_group_dict
        self.example_group = self.group_dict['P4']

    def test_lift_shape(self):
        signal_on_grid = tf.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            signal_on_group = conv_layer(signal_on_grid)
            self.assertEqual(signal_on_group.shape, self.shape[:-1] + (group.order, self.filters))

    def test_lift_shape_subgroup(self):
        signal_on_grid = tf.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
                conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv',
                                       subgroup=subgroup_name)

                signal_on_group = conv_layer(signal_on_grid)
                self.assertEqual(signal_on_group.shape, self.shape[:-1] + (subgroup.order, self.filters))

    def test_gc_shape(self):
        for group in self.group_dict.values():
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            new_signal = conv_layer(signal_on_group)
            self.assertEqual(new_signal.shape, signal_on_group.shape[:-1] + (self.filters,))

    def test_gc_shape_subgroup(self):
        for group in self.group_dict.values():
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
                conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv',
                                       subgroup=subgroup_name)

                new_signal = conv_layer(signal_on_group)
                self.assertEqual(new_signal.shape, self.shape[:-1] + (subgroup.order, self.filters))

    def test_lift_equiv(self):
        signal_on_grid = tf.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            conv_layer(signal_on_grid)
            conv_layer.bias = 1 + tf.random.normal(shape=conv_layer.bias.shape)
            equiv_diff = test_equivariance(
                conv_layer, signal_on_grid, spatial_axes=self.spatial_axes, group_axis=self.group_axis,
                domain_group=None)
            self.assertAllLess(equiv_diff, 1e-4)

    def test_lift_equiv_subgroup(self):
        signal_on_grid = tf.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv',
                                       subgroup=subgroup_name)
                conv_layer(signal_on_grid)
                conv_layer.bias = 1 + tf.random.normal(shape=conv_layer.bias.shape)
                equiv_diff = test_equivariance(
                    conv_layer, signal_on_grid, spatial_axes=self.spatial_axes, group_axis=self.group_axis,
                    acting_group=subgroup_name, target_group=subgroup_name, domain_group=None)
                self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv(self):
        for group in self.group_dict.values():
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            equiv_diff = test_equivariance(
                conv_layer, signal_on_group, group_axis=self.group_axis, spatial_axes=self.spatial_axes)
            self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv_subgroup(self):
        for group in self.group_dict.values():
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            for subgroup_name in group.subgroup.keys():
                conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv',
                                       subgroup=subgroup_name)
                equiv_diff = test_equivariance(
                    conv_layer, signal_on_group, group_axis=self.group_axis, spatial_axes=self.spatial_axes,
                    acting_group=subgroup_name, target_group=subgroup_name)

                self.assertAllLess(equiv_diff, 1e-4)

    def test_padding_equiv(self):
        for padding in ['same_equiv', 'valid_equiv']:
            for strides in [3, 5]:
                group = self.example_group
                signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
                conv_layer = self.conv(group=group, kernel_size=3, strides=strides, filters=self.filters, padding=padding)
                equiv_diff = test_equivariance(
                    conv_layer, signal_on_group, group_axis=self.group_axis, spatial_axes=self.spatial_axes)
                self.assertAllLess(equiv_diff, 1e-4)


class TestGroupConv2DTranspose(TestCase):
    def __init__(self, tests):
        super().__init__(tests)
        self.shape = (11, 7, 7, 3)
        self.spatial_axes = (1, 2)
        self.group_axis = 3  # for signals on group, not present in shape above
        self.filters = 5
        self.conv = GroupConv2DTranspose
        self.group_dict = wallpaper_group_dict
        self.example_group = self.group_dict['P4']

    def test_lift_shape(self):
        signal_on_grid = tf.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            signal_on_group = conv_layer(signal_on_grid)
            self.assertEqual(signal_on_group.shape, self.shape[:-1] + (group.order, self.filters))

    def test_gc_shape(self):
        for group in self.group_dict.values():
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            new_signal = conv_layer(signal_on_group)
            self.assertEqual(new_signal.shape, signal_on_group.shape[:-1] + (self.filters,))

    def test_lift_equiv(self):
        signal_on_grid = tf.random.normal(shape=self.shape, seed=42)
        for group in self.group_dict.values():
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            conv_layer(signal_on_grid)
            conv_layer.bias = 1 + tf.random.normal(shape=conv_layer.bias.shape)
            equiv_diff = test_equivariance(
                conv_layer, signal_on_grid, spatial_axes=self.spatial_axes, domain_group=None,
                group_axis=self.group_axis)
            self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_shape_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup_order = len(group.subgroup[subgroup_name])
                signal_on_subgroup = tf.random.normal(shape=self.shape[:-1] + (subgroup_order, self.shape[-1]), seed=42)
                conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv',
                                       subgroup=subgroup_name)
                signal_on_group = conv_layer(signal_on_subgroup)
                self.assertEqual(signal_on_group.shape, self.shape[:-1] + (group.order, self.filters))

    def test_gc_equiv(self):
        for group in self.group_dict.values():
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv')
            equiv_diff = test_equivariance(
                conv_layer, signal_on_group, group_axis=self.group_axis, spatial_axes=self.spatial_axes)
            self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
                signal_on_subgroup = tf.random.normal(shape=self.shape[:-1] + (subgroup.order, self.shape[-1]), seed=42)
                conv_layer = self.conv(group=group, kernel_size=3, filters=self.filters, padding='same_equiv',
                                       subgroup=subgroup_name)
                equiv_diff = test_equivariance(
                    conv_layer, signal_on_subgroup, group_axis=self.group_axis, spatial_axes=self.spatial_axes,
                    domain_group=subgroup_name, target_group=group.name, acting_group=subgroup_name)

                self.assertAllLess(equiv_diff, 1e-4)

    def test_padding_equiv_valid(self):
        padding = 'valid_equiv'
        for strides in [3, 5, 7]:
            group = self.example_group
            signal_on_group = tf.random.normal(shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42)
            conv_layer = self.conv(group=group, kernel_size=strides, strides=strides, filters=self.filters, padding=padding)
            equiv_diff = test_equivariance(
                conv_layer, signal_on_group, group_axis=self.group_axis, spatial_axes=self.spatial_axes)
            self.assertAllLess(equiv_diff, 1e-4)


if __name__ == "__main__":
    tf.test.main()
