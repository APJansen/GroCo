from tensorflow.keras.layers import Conv2D
import tensorflow as tf
from functools import partial
from groco.layers import GroupConvTransforms
from groco.groups import wallpaper_groups


class GroupConv2D(Conv2D):
    """
    Group convolution layer built on Keras's Conv2D layer.
    Additional arguments:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - subgroup: name of subgroup to act with (analog of a stride in the group direction), defaults to None,
                meaning full group is acted with
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    Expected input shape: (batch, height, width, group.order, in_channels), with the group.order axis omitted for the
    first layer (the lifting convolution).
    """
    def __init__(self, group, kernel_size, dimensions=2, allow_non_equivariance: bool = False, subgroup=None, **kwargs):
        self.group_transforms = GroupConvTransforms(
            allow_non_equivariance=allow_non_equivariance, kernel_size=kernel_size, dimensions=dimensions,
            group=group, subgroup=subgroup, **kwargs)
        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup
        if 'padding' in kwargs and kwargs['padding'].endswith('_equiv'):
            kwargs['padding'] = kwargs['padding'][:-6]
        super().__init__(kernel_size=kernel_size, **kwargs)
        self.group_valued_input = None

    def call(self, inputs):
        inputs = self.group_transforms.merge_group_axis_and_pad(inputs)
        kernel = self.group_transforms.transform_kernel(self.kernel)
        bias = self.group_transforms.repeat_bias(self.bias)

        outputs = self._conv_call(inputs, kernel, bias)

        return self.group_transforms.restore_group_axis(outputs)

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.group_valued_input = self.group_transforms.group_valued_input
        super().build(reshaped_input)
        self.group_transforms.compute_indices(input_shape, self.kernel, self.bias)
        if self.group_valued_input:
            self.input_spec.axes = {self._get_channel_axis(): input_shape[self.group_transforms.channels_axis]}

    def get_config(self):
        config = super().get_config()
        config.update(self.group_transforms.get_config())
        return config

    def _conv_call(self, reshaped_inputs, reshaped_kernel, reshaped_bias):
        """
        This is the original call method, with replacements:
        - self.kernel -> reshaped_kernel (argument)
        - self.bias -> reshaped_bias (argument)
        - self.filters -> filters_multiplied = self.filters * self.group.order
        """
        input_shape = reshaped_inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            reshaped_inputs = tf.pad(reshaped_inputs, self._compute_causal_padding(reshaped_inputs))

        outputs = self._convolution_op(reshaped_inputs, reshaped_kernel)  # had to add underscore to make it work

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                filters_multiplied = self.filters * self.group.order
                bias = tf.reshape(reshaped_bias, (1, filters_multiplied, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(o, reshaped_bias, data_format=self._tf_data_format)

                    outputs = tf.keras.utils.conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = tf.nn.bias_add(
                        outputs, reshaped_bias, data_format=self._tf_data_format)

        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self._compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        """
        Multiply channels with group order.
        Note the layer's output shape is changed later to have a separate group index, this is to be
        consistent with the conv2d_call method.
        """
        return self.group_transforms.multiply_channels(super().compute_output_shape(input_shape))


P4MConv2D = partial(GroupConv2D, group=wallpaper_groups.P4M)
P4Conv2D = partial(GroupConv2D, group=wallpaper_groups.P4)
P2MMConv2D = partial(GroupConv2D, group=wallpaper_groups.P2MM)
PMhConv2D = partial(GroupConv2D, group=wallpaper_groups.PMh)
PMwConv2D = partial(GroupConv2D, group=wallpaper_groups.PMw)
P2Conv2D = partial(GroupConv2D, group=wallpaper_groups.P2)
P1Conv2D = partial(GroupConv2D, group=wallpaper_groups.P1)
