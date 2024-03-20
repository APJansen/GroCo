import keras
from keras import ops

from geqco.groups import space_group_dict, wallpaper_group_dict
from geqco.layers import (
    GlobalGroupMaxPooling2D,
    GlobalGroupMaxPooling3D,
    GroupMaxPooling2D,
    GroupMaxPooling3D,
)
from geqco.utils import check_equivariance
from tests.custom_testcase import KerasTestCase as TestCase


def get_attributes(dimension: int, is_global: bool = False):
    if dimension == 2:
        pool_layer = GroupMaxPooling2D if not is_global else GlobalGroupMaxPooling2D
        group_dict = wallpaper_group_dict
    elif dimension == 3:
        pool_layer = GroupMaxPooling3D if not is_global else GlobalGroupMaxPooling3D
        group_dict = space_group_dict

    return pool_layer, group_dict


class TestGroupPoolingBase:
    dimension = None

    def __init__(self, tests):
        super().__init__(tests)
        self.xsize = 8
        self.batch_size = 2
        self.input_features = 3
        self.shape = (self.batch_size,) + (self.xsize,) * self.dimension + (self.input_features,)
        self.pooled_shape = (
            (self.batch_size,) + (self.xsize // 2,) * self.dimension + (self.input_features,)
        )
        self.spatial_axes = tuple(range(1, self.dimension + 1))
        self.group_axis = self.dimension + 1

        self.pool, self.group_dict = get_attributes(self.dimension)

    def test_pool_shape(self):
        for group in self.group_dict.values():
            signal_on_group = keras.random.normal(
                shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
            )
            pool_layer = self.pool(group=group, pool_size=2, strides=2, padding="same_equiv")
            pooled_signal = pool_layer(signal_on_group)
            self.assertEqual(
                pooled_signal.shape, self.pooled_shape[:-1] + (group.order, self.shape[-1])
            )

    def test_pool_shape_subgroup(self):
        for group in self.group_dict.values():
            signal_on_group = keras.random.normal(
                shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
            )
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
                pool_layer = self.pool(
                    group=group,
                    pool_size=2,
                    strides=2,
                    padding="same_equiv",
                    subgroup=subgroup_name,
                )
                pooled_signal = pool_layer(signal_on_group)
                self.assertEqual(
                    pooled_signal.shape, self.pooled_shape[:-1] + (subgroup.order, self.shape[-1])
                )

    def test_pool_equiv(self):
        for group in self.group_dict.values():
            signal_on_group = keras.random.normal(
                shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
            )
            pool_layer = self.pool(group=group, pool_size=2, strides=2, padding="same_equiv")
            equiv_diff = check_equivariance(
                pool_layer,
                signal_on_group,
                group_axis=self.group_axis,
                spatial_axes=self.spatial_axes,
            )

            self.assertAllLess(equiv_diff, 1e-4)

    def test_pool_equiv_subgroup(self):
        for group in self.group_dict.values():
            for subgroup_name in group.subgroup.keys():
                signal_on_group = keras.random.normal(
                    shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
                )
                pool_layer = self.pool(
                    group=group,
                    pool_size=2,
                    strides=2,
                    padding="same_equiv",
                    subgroup=subgroup_name,
                )
                equiv_diff = check_equivariance(
                    pool_layer,
                    signal_on_group,
                    group_axis=self.group_axis,
                    acting_group=subgroup_name,
                    spatial_axes=self.spatial_axes,
                    target_group=subgroup_name,
                )

                self.assertAllLess(equiv_diff, 1e-4)


class TestGlobalGroupPoolingBase:
    dimension = None

    def __init__(self, tests):
        super().__init__(tests)
        self.xsize = 8
        self.batch_size = 2
        self.input_features = 3
        self.shape = (self.batch_size,) + (self.xsize,) * self.dimension + (self.input_features,)
        self.pooled_shape = (self.batch_size,) + (self.input_features,)
        self.spatial_axes = tuple(range(1, self.dimension + 1))
        self.group_axis = self.dimension + 1

        self.pool, self.group_dict = get_attributes(self.dimension, is_global=True)

    def test_pool_shape(self):
        for group in self.group_dict.values():
            signal_on_group = keras.random.normal(
                shape=self.shape[:-1] + (group.order, self.shape[-1]), seed=42
            )
            pool_layer = self.pool()
            pooled_signal = pool_layer(signal_on_group)
            self.assertEqual(pooled_signal.shape, self.pooled_shape)


class TestGroupPooling2D(TestGroupPoolingBase, TestCase):
    dimension = 2


class TestGroupPooling3D(TestGroupPoolingBase, TestCase):
    dimension = 3


class TestGlobalGroupPooling2D(TestGlobalGroupPoolingBase, TestCase):
    dimension = 2


class TestGlobalGroupPooling3D(TestGlobalGroupPoolingBase, TestCase):
    dimension = 3
