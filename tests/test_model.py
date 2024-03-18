import keras
from keras import ops
from keras.layers import Input
from keras.models import Sequential

from groco.groups import wallpaper_group_dict
from groco.layers import (
    GlobalGroupMaxPooling2D,
    GroupConv2D,
    GroupConv2DTranspose,
    GroupMaxPooling2D,
)
from groco.utils import check_equivariance
from tests.custom_testcase import KerasTestCase as TestCase


class TestModel(TestCase):
    def __init__(self, tests):
        super().__init__(tests)
        self.example_group = wallpaper_group_dict["P4"]
        self.input_shape_group = (1, 8, 8, self.example_group.order, 3)
        self.signal_on_group = keras.random.normal(shape=self.input_shape_group, seed=42)
        self.group_axis = 3
        self.padding = "same_equiv"
        self.model = Sequential(
            [
                GroupConv2D(
                    group=self.example_group,
                    kernel_size=3,
                    filters=3,
                    padding=self.padding,
                ),
                GroupMaxPooling2D(group=self.example_group, pool_size=2),
                GroupConv2D(
                    group=self.example_group,
                    kernel_size=3,
                    filters=3,
                    padding=self.padding,
                ),
                GroupMaxPooling2D(group=self.example_group, pool_size=2),
                GroupConv2DTranspose(
                    group=self.example_group,
                    kernel_size=3,
                    filters=3,
                    padding=self.padding,
                ),
            ]
        )

    def test_model_shape(self):
        output = self.model(self.signal_on_group)
        self.assertEqual(output.shape, (1, 2, 2, self.example_group.order, 3))

    def test_model_equivariance(self):
        equiv_diff = check_equivariance(
            layer=self.model,
            signal=self.signal_on_group,
            group=self.example_group,
            group_axis=self.group_axis,
        )
        self.assertAllLess(equiv_diff, 1e-4)
