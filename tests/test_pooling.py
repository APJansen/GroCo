import keras
from keras import ops
from pytest.unittest import TestCase

from groco.groups import space_group_dict, wallpaper_group_dict
from groco.layers import GroupMaxPooling2D, GroupMaxPooling3D
from groco.utils import check_equivariance


class TestGroupPooling2D(TestCase):
    def __init__(self, tests):
        super().__init__(tests)
        self.shape = (9, 8, 8, 3)
        self.pooled_shape = (9, 4, 4, 3)
        self.spatial_axes = (1, 2)
        self.group_axis = 3
        self.pool = GroupMaxPooling2D
        self.group_dict = wallpaper_group_dict

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


class TestGroupPooling3D(TestCase):
    def __init__(self, tests):
        super().__init__(tests)
        self.shape = (9, 8, 8, 8, 3)
        self.pooled_shape = (9, 4, 4, 4, 3)
        self.spatial_axes = (1, 2, 3)
        self.group_axis = 4
        self.pool = GroupMaxPooling3D
        self.group_dict = space_group_dict

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
            for subgroup_name in group.subgroup.keys():
                subgroup = self.group_dict[subgroup_name]
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
