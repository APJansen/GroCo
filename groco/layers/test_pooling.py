from tensorflow.test import TestCase
from groco.groups.wallpaper_groups import group_dict
from groco.layers.pooling import GroupMaxPooling2D
from groco.utils.test_equivariance import test_equivariance
import tensorflow as tf


class TestGroupPooling2D(TestCase):

    def test_pool_shape(self):
        for group in group_dict.values():
            signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
            pool_layer = GroupMaxPooling2D(group=group, pool_size=2, strides=2, padding='same_equiv')
            pooled_signal = pool_layer(signal_on_group)
            self.assertEqual(pooled_signal.shape, (10, 14, 14, group.order, 3))

    def test_pool_shape_subgroup(self):
        for group in group_dict.values():
            for subgroup_name in group.subgroup.keys():
                subgroup = group_dict[subgroup_name]
                signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
                pool_layer = GroupMaxPooling2D(group=group, pool_size=2, strides=2, padding='same_equiv',
                                               subgroup=subgroup_name)
                pooled_signal = pool_layer(signal_on_group)
                self.assertEqual(pooled_signal.shape, (10, 14, 14, subgroup.order, 3))

    def test_pool_equiv(self):
        for group in group_dict.values():
            signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
            pool_layer = GroupMaxPooling2D(group=group, pool_size=2, strides=2, padding='same_equiv')
            equiv_diff = test_equivariance(pool_layer, signal_on_group, group_axis=3)

            self.assertAllLess(equiv_diff, 1e-4)

    def test_pool_equiv_subgroup(self):
        for group in group_dict.values():
            for subgroup_name in group.subgroup.keys():
                signal_on_group = tf.random.normal(shape=(10, 28, 28, group.order, 3), seed=42)
                pool_layer = GroupMaxPooling2D(group=group, pool_size=2, strides=2, padding='same_equiv',
                                               subgroup=subgroup_name)
                equiv_diff = test_equivariance(pool_layer, signal_on_group, group_axis=3,
                                               subgroup=subgroup_name)

                self.assertAllLess(equiv_diff, 1e-4)

tf.test.main()
