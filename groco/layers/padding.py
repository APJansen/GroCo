from keras import ops
from keras.layers import Layer


class EquivariantPadding(Layer):
    """
    Base padding layer.

    This is meant to be used internally by GroupConv and GroupPooling layers.
    The padding done here is in addition to any potential padding done in those layers.

    Specifically what it imposes is that the origin of the output grid coincides with the
    origin of the input grid.
    For this to be true, two conditions need to be met:
    - the total padding should be even, so it can be applied evenly on both sides
    - the sampling shouldn't stop before the end of the image, resulting in the condition:
     (input_size + padding - kernel_size) % stride == 0

    The options `valid_equiv` and `same_equiv` for the padding argument will do the minimal
    amount of extra padding on top of their usual counterparts to maintain equivariance.

    Args:
        kernel_size: int or tuple of int, the size of the kernel.
        dimensions: int, the number of spatial dimensions.
        strides: int or tuple of int, the stride of the convolution.
        padding: string, one of 'valid_equiv', 'same_equiv', 'valid', 'same'.
        allow_non_equivariance: bool, whether to allow non-equivariant padding.
        data_format: string, one of 'channels_last', 'channels_first'.
        transpose: bool, whether the layer is a transposed convolution.
    """

    def __init__(
        self,
        kernel_size,
        dimensions: int,
        strides=1,
        padding="valid_equiv",
        allow_non_equivariance=False,
        data_format="channels_last",
        transpose=False,
        **kwargs,
    ):
        self.padding_option = self.format_padding_option(padding)
        self.built_in_padding_option = padding[:-6] if padding.endswith("_equiv") else padding

        self.allow_non_equivariance = allow_non_equivariance
        self.dimensions = dimensions
        self.strides = (
            strides
            if isinstance(strides, tuple)
            else tuple(strides for _ in range(self.dimensions))
        )
        kernel_size_tuple = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else tuple(kernel_size for _ in range(self.dimensions))
        )
        self.kernel_sizes = ops.cast(kernel_size_tuple, "int32")
        self.data_format = data_format
        self.transpose = transpose
        # set during build
        self.equivariant_padding = None
        self.needs_padding = None

        super().__init__()

    def call(self, inputs):
        if self.needs_padding:
            return ops.pad(inputs, self.equivariant_padding)
        else:
            return inputs

    def build(self, input_shape):
        if self.padding_option in ["SAME", "VALID"] and self.allow_non_equivariance:
            self.needs_padding = False
            return

        if self.data_format == "channels_last":
            spatial_shape = ops.cast(input_shape[1 : 1 + self.dimensions], "int32")
        else:
            spatial_shape = ops.cast(input_shape[2:], "int32")
        extra_padding, total_padding = self.compute_equivariant_padding(spatial_shape)

        self.needs_padding = ops.any(extra_padding != 0)
        if not self.needs_padding:
            return
        elif self.padding_option in ["SAME", "VALID"]:
            raise self.non_equivariant_error(spatial_shape)

        # if the stride is even the padding can still be odd, no padding will maintain equivariance
        if ops.any(total_padding % 2 != 0):
            raise self.padding_parity_error(spatial_shape)

        self.equivariant_padding = self.split_padding(extra_padding)

    def compute_equivariant_padding(self, spatial_shape):
        built_in_padding = self.compute_built_in_padding(spatial_shape)
        extra_padding = self.compute_extra_padding(spatial_shape, built_in_padding)
        return extra_padding

    def compute_built_in_padding(self, spatial_shape):
        if self.padding_option in ["SAME", "SAME_EQUIV"]:
            return ops.relu(self.kernel_sizes - self.strides + (-spatial_shape % self.strides))
        else:
            if self.transpose:
                return ops.relu(self.strides - self.kernel_sizes)
            else:
                return ops.zeros_like(self.kernel_sizes)

    def compute_extra_padding(self, spatial_shape, built_in_padding):
        extra_padding = (-(spatial_shape + built_in_padding - self.kernel_sizes)) % self.strides
        # add stride where necessary to make total padding even
        total_padding = extra_padding + built_in_padding
        extra_padding = extra_padding + (total_padding % 2) * self.strides
        total_padding = extra_padding + built_in_padding
        return extra_padding, total_padding

    def split_padding(self, paddings):
        # add 0 at both ends, not to pad the batch and channel axes
        if self.data_format == "channels_last":
            paddings = ops.pad(paddings, [[1, 1]])
        else:
            paddings = ops.pad(paddings, [[2, 0]])

        # paddings may be odd if the 'same' paddings are too, must make sure to do this opposite to 'same' padding
        # to make the total padding the same on both sides
        pads_after = paddings // 2
        pads_before = paddings - pads_after

        return ops.concatenate(
            [ops.expand_dims(pads_before, axis=1), ops.expand_dims(pads_after, axis=1)], axis=1
        )

    def get_config(self):
        return {
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "allow_non_equivariance": self.allow_non_equivariance,
            "dimensions": self.dimensions,
        }

    @staticmethod
    def format_padding_option(padding_option):
        if not isinstance(padding_option, str):
            raise TypeError(f"padding option should be a string, received {padding_option}.")
        capitalized = padding_option.upper()
        if capitalized not in ["SAME", "VALID", "SAME_EQUIV", "VALID_EQUIV"]:
            raise TypeError(
                f"padding option should be one of ['same', 'valid', 'same_equiv', 'valid_equiv'], received {padding_option}."
            )
        return capitalized

    def padding_parity_error(self, spatial_shape):
        message = (
            "Unable to find padding that maintains equivariance.\n"
            "Constraints (input_shape + padding - kernel_size) % stride == 0 and "
            "padding being even not compatible.\n" + "Values found: \n"
        )
        message += self.configuration_string(spatial_shape)
        return ValueError(message)

    def non_equivariant_error(self, spatial_shape):
        message = (
            "Current configuration will spoil equivariance.\n"
            "Changing to `padding='valid_equiv'` or `padding='same_equiv'` "
            "will add the minimal amount of extra padding to restore equivariance.\n"
            "If you insist on using the current padding, use `allow_non_equivariance=True`.\n"
        )
        message += self.configuration_string(spatial_shape)
        return ValueError(message)

    def configuration_string(self, spatial_shape):
        message = (
            f"input_shape (spatial): {tuple(spatial_shape)},\n"
            f"kernel/pool size: {tuple(self.kernel_sizes)},\n"
            f"strides: {tuple(self.strides)},\n"
        )
        if self.padding_option == "SAME_EQUIV":
            same_padding = self.compute_same_padding(spatial_shape)
            message += f"built-in same padding: {tuple(same_padding)}"
        return message
