from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Layer
import tensorflow as tf
from groco.layers import GroupConvTransforms


class GroupPooling2D(Layer):
    """
    Layer for pooling on signals on a group. Checks and restores equivariance in the same way as GroupConv2D.

    Only meant to be subclassed by GroupMaxPooling2D and GroupAveragePooling2D.
    """
    def __init__(self, group, pool_type: str, allow_non_equivariance: bool = False, pool_size=(2, 2), subgroup=None,
                 **kwargs):
        self.group_transforms = GroupConvTransforms(
            allow_non_equivariance=allow_non_equivariance, kernel_size=pool_size, dimensions=2,
            group=group, subgroup=subgroup, **kwargs)
        kwargs['padding'] = self.group_transforms.built_in_padding_option
        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup

        self.pool_type = pool_type
        pool_layer = MaxPooling2D if self.pool_type == 'max' else AveragePooling2D
        self.pooling = pool_layer(pool_size=pool_size, **kwargs)

        super().__init__()

        self.pooling_indices = None  # created during build

    def call(self, inputs):
        inputs = self.group_transforms.merge_group_axis_and_pad(inputs)
        outputs = self.pooling(inputs)
        outputs = self.group_transforms.restore_group_axis(outputs, subgroup=False)

        return self._subgroup_pooling(outputs)

    def _subgroup_pooling(self, outputs):
        outputs = tf.gather(outputs, axis=3, indices=self.pooling_indices)
        if self.pool_type == 'max':
            outputs = tf.reduce_max(outputs, axis=4)
        else:
            outputs = tf.reduce_mean(outputs, axis=4)
        return outputs

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.pooling.build(reshaped_input)
        self.pooling_indices = self._create_pooling_indices()

    def _create_pooling_indices(self):
        indices = tf.gather(self.group.composition, axis=1, indices=self.group.cosets[self.subgroup.name])
        subgroup_indices = self.group.subgroup[self.subgroup.name]
        indices = tf.gather(indices, axis=0, indices=subgroup_indices)

        return indices

    def get_config(self):
        config = self.pooling.get_config()
        config.update(self.group_transforms.get_config())
        return config


class GroupMaxPooling2D(GroupPooling2D):
    """
    Layer for pooling on signals on a group. Allows for pooling on the grid as usual,
    but also on subgroups of the point group.
    Checks and restores equivariance in the same way as GroupConv2D.

    Additional arguments to MaxPooling2D:
    group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups.
    padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
    to maintain equivariance.
    allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
    non-equivariant padding.
    subgroup: defaults to None meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', **kwargs)


class GroupAveragePooling2D(GroupPooling2D):
    """
    Layer for pooling on signals on a group. Allows for pooling on the grid as usual,
    but also on subgroups of the point group.
    Checks and restores equivariance in the same way as GroupConv2D.

    Additional arguments to AveragePooling2D:
    group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups.
    padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
    to maintain equivariance.
    allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
    non-equivariant padding.
    subgroup: defaults to None meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', **kwargs)
