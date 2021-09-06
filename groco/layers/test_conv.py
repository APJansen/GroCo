from tensorflow.test import TestCase
from groco.groups.wallpaper_groups import group_dict
from groco.layers.conv import GroupConv2D
from groco.utils.test_equivariance import test_equivariance
import tensorflow as tf


class TestGroupConv2D(TestCase):

    def test_lift_shape(self):
        filters = 5
        signal_on_grid = tf.random.normal(shape=(10, 28, 28, 3), seed=42)
        for group in group_dict.values():
            conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv')
            signal_on_group = conv_layer(signal_on_grid)
            self.assertEqual(signal_on_group.shape, signal_on_grid.shape[:3] + (group.order, filters))

    def test_lift_shape_subgroup(self):
        filters = 5
        signal_on_grid = tf.random.normal(shape=(10, 28, 28, 3), seed=42)
        for group in group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = group_dict[subgroup_name]
                conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv',
                                         subgroup=subgroup_name)

                signal_on_group = conv_layer(signal_on_grid)
                self.assertEqual(signal_on_group.shape, signal_on_grid.shape[:3] + (subgroup.order, filters))

    def test_gc_shape(self):
        filters = 5
        for group in group_dict.values():
            signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
            conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv')
            new_signal = conv_layer(signal_on_group)
            self.assertEqual(new_signal.shape, signal_on_group.shape[:-1] + (filters, ))

    def test_gc_shape_subgroup(self):
        filters = 5
        for group in group_dict.values():
            signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
            for subgroup_name in group.subgroup.keys():
                subgroup = group_dict[subgroup_name]
                conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv',
                                         subgroup=subgroup_name)

                new_signal = conv_layer(signal_on_group)
                self.assertEqual(new_signal.shape, signal_on_group.shape[:3] + (subgroup.order, filters))

    def test_lift_equiv(self):
        filters = 5
        signal_on_grid = tf.random.normal(shape=(10, 28, 28, 3), seed=42)
        for group in group_dict.values():
            conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv')
            conv_layer(signal_on_grid)
            conv_layer.bias = 1 + tf.random.normal(shape=conv_layer.bias.shape)
            equiv_diff = test_equivariance(conv_layer, signal_on_grid)
            self.assertAllLess(equiv_diff, 1e-4)

    def test_lift_equiv_subgroup(self):
        filters = 5
        signal_on_grid = tf.random.normal(shape=(10, 28, 28, 3), seed=42)
        for group in group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = group_dict[subgroup_name]
                conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv',
                                         subgroup=subgroup_name)
                conv_layer(signal_on_grid)
                conv_layer.bias = 1 + tf.random.normal(shape=conv_layer.bias.shape)
                equiv_diff = test_equivariance(conv_layer, signal_on_grid, subgroup=subgroup_name)
                self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv(self):
        filters = 5
        for group in group_dict.values():
            signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
            conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv')
            equiv_diff = test_equivariance(conv_layer, signal_on_group, group_axis=3)

            self.assertAllLess(equiv_diff, 1e-4)

    def test_gc_equiv_subgroup(self):
        filters = 5
        for group in group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = group_dict[subgroup_name]
                signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
                conv_layer = GroupConv2D(group=group, kernel_size=3, filters=filters, padding='same_equiv',
                                         subgroup=subgroup_name)
                equiv_diff = test_equivariance(conv_layer, signal_on_group, group_axis=3, subgroup=subgroup_name)

                self.assertAllLess(equiv_diff, 1e-4)





tf.test.main()