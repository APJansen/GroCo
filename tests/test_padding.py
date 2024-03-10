import keras
from keras import ops
import tensorflow as tf
from tensorflow.test import TestCase

from groco.layers import EquivariantPadding


class TestEquivariantPadding(TestCase):
    def test_valid_equiv_1d(self):
        # padding in this case  (-(spatial_shape - kernel_size)) % stride = 2
        layer = EquivariantPadding(kernel_size=2, dimensions=1, strides=3, padding="valid_equiv")
        inputs = keras.random.normal(shape=(1, 30, 3))
        output = layer(inputs)
        self.assertEqual(output.shape[1], inputs.shape[1] + 2)

    def test_valid_equiv_2d(self):
        # padding in this case  (-(spatial_shape - kernel_size)) % stride = 2, 1; but add 3 to make even: 2, 4
        layer = EquivariantPadding(kernel_size=2, dimensions=2, strides=3, padding="valid_equiv")
        inputs = keras.random.normal(shape=(1, 30, 31, 3))
        outputs = layer(inputs)
        output_shape = tuple(s + p for s, p in zip(inputs.shape[1:3], (2, 4)))
        self.assertAllEqual(outputs.shape[1:3], output_shape)

    def test_valid_equiv_3d(self):
        # padding in this case  (-(spatial_shape - kernel_size)) % stride = 2, 4, 0
        layer = EquivariantPadding(kernel_size=2, dimensions=3, strides=3, padding="valid_equiv")
        inputs = keras.random.normal(shape=(1, 30, 31, 32, 3))
        output = layer(inputs)
        output_shape = tuple(s + p for s, p in zip(inputs.shape[1:4], (2, 4, 0)))
        self.assertAllEqual(output.shape[1:4], output_shape)

    def test_same_equiv_1d(self):
        # same padding here kernel - stride + (-spatial_shape % stride)) = 2 - 3 = -1 -> 0
        # padding in this case  (-(spatial_shape + padding - kernel_size)) % stride = 2
        layer = EquivariantPadding(kernel_size=2, dimensions=1, strides=3, padding="same_equiv")
        inputs = keras.random.normal(shape=(1, 30, 3))
        output = layer(inputs)
        self.assertEqual(output.shape[1], inputs.shape[1] + 2)

    def test_same_equiv_2d(self):
        # same padding here kernel - stride + (-spatial_shape % stride)) = 2 - 3 + (0, 2) = -1, 1  -> 0, 1
        # padding in this case  (-(spatial_shape + padding - kernel_size)) % stride = 2, 0; but add 3 to make even: 2, 3
        layer = EquivariantPadding(kernel_size=2, dimensions=2, strides=3, padding="same_equiv")
        inputs = keras.random.normal(shape=(1, 30, 31, 3))
        output = layer(inputs)
        output_shape = tuple(s + p for s, p in zip(inputs.shape[1:3], (2, 3)))
        self.assertAllEqual(output.shape[1:3], output_shape)

    def test_same_equiv_3d(self):
        # same padding here kernel - stride + (-spatial_shape % stride)) = 2 - 3 + (0, 2, 1) = -1, 1, 0  -> 0, 1, 0
        # padding in this case  (-(spatial_shape + padding - kernel_size)) % stride = 2, 0, 0; but add 3 to make even: 2, 3, 0
        layer = EquivariantPadding(kernel_size=2, dimensions=3, strides=3, padding="same_equiv")
        inputs = keras.random.normal(shape=(1, 30, 31, 32, 3))
        outputs = layer(inputs)
        output_shape = tuple(s + p for s, p in zip(inputs.shape[1:4], (2, 3, 0)))
        self.assertAllEqual(outputs.shape[1:4], output_shape)

    def test_channels_first(self):
        layer = EquivariantPadding(
            kernel_size=2,
            dimensions=2,
            strides=3,
            pading="valid_equiv",
            data_format="channels_first",
        )
        inputs = keras.random.normal(shape=(1, 3, 30, 31))
        outputs = layer(inputs)
        output_shape = tuple(s + p for s, p in zip(inputs.shape[2:], (2, 4)))
        self.assertAllEqual(outputs.shape[2:], output_shape)


if __name__ == "__main__":
    tf.test.main()
