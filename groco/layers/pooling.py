from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Layer
import tensorflow as tf
from groco.groups import Group, wallpaper_groups
from groco.layers import EquivariantPadding


class GroupPooling2D(Layer):
    """
    Layer for pooling on signals on a group. Checks and restores equivariance in the same way as GroupConv2D.

    Only meant to be subclassed by GroupMaxPooling2D and GroupAveragePooling2D.
    """
    def __init__(self, group, pool_type: str, allow_non_equivariance: bool = False, pool_size=(2, 2), subgroup=None,
                 **kwargs):
        self.group = group if isinstance(group, Group) else wallpaper_groups.group_dict[group]
        self.subgroup_name = self.group.name if subgroup is None else subgroup
        self.subgroup = self.group if subgroup is None else wallpaper_groups.group_dict[subgroup]

        self.equivariant_padding = EquivariantPadding(allow_non_equivariance=allow_non_equivariance,
                                                      kernel_size=pool_size, **kwargs)
        if 'padding' in kwargs and kwargs['padding'].endswith('_equiv'):
            kwargs['padding'] = kwargs['padding'][:-6]

        self.pool_type = pool_type
        if pool_type == 'max':
            self.pooling = MaxPooling2D(pool_size=pool_size, **kwargs)
        else:
            self.pooling = AveragePooling2D(pool_size=pool_size, **kwargs)

        super().__init__()

        self.pooling_indices = None # created during build

    def call(self, inputs):
        inputs = self._merge_group_axis(inputs)
        inputs = self.equivariant_padding(inputs)
        outputs = self.pooling(inputs)
        outputs = self._restore_group_axis(outputs)

        outputs = self._subgroup_pooling(outputs)

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

    def _subgroup_pooling(self, outputs):
        outputs = tf.gather(outputs, axis=3, indices=self.pooling_indices)
        if self.pool_type == 'max':
            outputs = tf.reduce_max(outputs, axis=4)
        else:
            outputs = tf.reduce_mean(outputs, axis=4)
        return outputs

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

        self.pooling_indices = self._create_pooling_indices()

    def _create_pooling_indices(self):
        indices = tf.gather(self.group.composition, axis=1, indices=self.group.cosets[self.subgroup_name])
        subgroup_indices = [self.group.inverses[i] for i in self.group.subgroup[self.subgroup_name]]
        indices = tf.gather(indices, axis=0, indices=subgroup_indices)

        return indices

    def get_config(self):
        config = self.pooling.get_config()
        config['group'] = self.group
        config['allow_non_equivariance'] = self.equivariant_padding.allow_non_equivariance
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
