from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Layer
import tensorflow as tf
from groco import groups
from groco.utils import EquivariantPadding

# was planning to subclass `Pooling2D` layer, but this is not meant to be exposed outside of Keras internals
# so a. not sure how to do it, and b. might not be a good idea?
# would remove the code duplication though

class GroupPooling2D(Layer):
    """
    Layer for pooling on signals on a group. Checks and restores equivariance in the same way as GroupConv2D.

    Only meant to be subclassed by GroupMaxPooling2D and GroupAveragePooling2D.
    """
    def __init__(self, group, pool_type: str, allow_non_equivariance: bool = False, pool_size=(2, 2), **kwargs):
        self.group = group if isinstance(group, groups.WallpaperGroup) else groups.group_dict[group]

        self.equivariant_padding = EquivariantPadding(allow_non_equivariance=allow_non_equivariance,
                                                      kernel_size=pool_size, **kwargs)
        if 'padding' in kwargs and kwargs['padding'].endswith('_equiv'):
            kwargs['padding'] = kwargs['padding'][:-6]

        if pool_type == 'max':
            self.pooling = MaxPooling2D(pool_size=pool_size, **kwargs)
        else:
            self.pooling = AveragePooling2D(pool_size=pool_size, **kwargs)

        super().__init__()

    def call(self, inputs):
        inputs = self._merge_group_axis(inputs)
        inputs = self.equivariant_padding(inputs)
        outputs = self.pooling(inputs)
        outputs = self._restore_group_axis(outputs)
        return outputs

    def _merge_group_axis(self, inputs):
        """
        If the input is a signal on the group, join the group axis with the channel axis.

        (batch, height, width, group_order, channels) -> (batch, height, width, group_order * channels)
        """
        batch, height, width, group_order, channels = inputs.shape
        batch = -1 if batch is None else batch
        return tf.reshape(inputs, (batch, height, width, group_order * channels))

    def _restore_group_axis(self, outputs):
        """
        Reshape the output of the pooling, splitting off the group index from the channel axis.

        (batch, height, width, group_order * channels) -> (batch, height, width, group_order, channels)
        """
        batch, height, width, channels = outputs.shape
        batch = -1 if batch is None else batch
        return tf.reshape(outputs, (batch, height, width, self.group.order, channels // self.group.order))

    def build(self, input_shape):
        """
        Merge the group axis with the channel axis.
        Then run the parent class's build.
        Run the EquivariantPadding layer's build on the merged input.
        """
        super().build(input_shape)
        (batch, height, width, group_order, channels) = input_shape
        assert group_order == self.group.order, f'Got input shape {input_shape}, expected {(batch, height, width, self.group.order, channels)}.'
        input_shape = (batch, height, width, channels * group_order)

        self.equivariant_padding.build(input_shape)
        self.pooling.build(input_shape)

    def get_config(self):
        config = self.pooling.get_config()
        config['group'] = self.group
        config['allow_non_equivariance'] = self.equivariant_padding.allow_non_equivariance
        return config


class GroupMaxPooling2D(GroupPooling2D):
    """
    Layer for pooling on signals on a group. Checks and restores equivariance in the same way as GroupConv2D.

    Additional arguments to MaxPooling2D:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', **kwargs)


class GroupAveragePooling2D(GroupPooling2D):
    """
    Layer for pooling on signals on a group. Checks and restores equivariance in the same way as GroupConv2D.

    Additional arguments to MaxPooling2D:
    - group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups
    - padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
               to maintain equivariance
    - allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
                              non-equivariant padding

    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', **kwargs)
