from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D
from tensorflow.keras.layers import Layer
from groco.layers import GroupTransforms
import tensorflow as tf


class GroupPooling(Layer):
    """
    Layer for pooling on signals on a group. Checks and restores equivariance in the same way as GroupConv2D.

    Only meant to be subclassed.
    """
    def __init__(self, group, dimensions: int, pool_type: str, allow_non_equivariance: bool = False, subgroup='',
                 pool_size=2, **kwargs):
        self.dimensions = dimensions
        self.group_transforms = GroupTransforms(
            allow_non_equivariance=allow_non_equivariance, kernel_size=pool_size, dimensions=dimensions,
            group=group, subgroup=subgroup, pooling=True, **kwargs)
        kwargs['padding'] = self.group_transforms.built_in_padding_option
        self.group = self.group_transforms.group
        self.subgroup = self.group_transforms.subgroup

        self.pool_type = pool_type
        if self.dimensions == 1:
            pool_layer = MaxPooling1D if self.pool_type == 'max' else AveragePooling1D
        elif self.dimensions == 2:
            pool_layer = MaxPooling2D if self.pool_type == 'max' else AveragePooling2D
        else:
            pool_layer = MaxPooling3D if self.pool_type == 'max' else AveragePooling3D
        self.pooling = pool_layer(pool_size=pool_size, **kwargs)

        super().__init__()

        self.pooling_indices = None  # created during build

    def call(self, inputs):
        inputs = self.group_transforms.subgroup_pooling(inputs, self.pool_type)
        inputs = self.group_transforms.merge_group_axis_and_pad(inputs)
        outputs = self.pooling(inputs)
        return self.group_transforms.restore_group_axis(outputs)

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.pooling.build(reshaped_input)
        self.pooling_indices = self.group_transforms.compute_pooling_indices()

    def get_config(self):
        config = self.pooling.get_config()
        config.update(self.group_transforms.get_config())
        return config


class GlobalGroupPooling(Layer):
    """
    Layer for global pooling on signals on a group.

    Only meant to be subclassed by GlobalGroupMaxPooling2D and GlobalGroupAveragePooling2D,
    and other dimensions.
    """
    def __init__(self, dimensions: int, pool_type: str, **kwargs):
        self.dimensions = dimensions
        self.pool_type = pool_type
        if self.dimensions == 1:
            pool_layer = GlobalMaxPooling1D if self.pool_type == 'max' else GlobalAveragePooling1D
        elif self.dimensions == 2:
            pool_layer = GlobalMaxPooling2D if self.pool_type == 'max' else GlobalAveragePooling2D
        else:
            pool_layer = GlobalMaxPooling3D if self.pool_type == 'max' else GlobalAveragePooling3D
        self.pooling = pool_layer(**kwargs)

        if 'data_format' in kwargs and kwargs['data_format'] == 'channels_first':
            self.group_axis = self.dimensions + 2
        else:
            self.group_axis = self.dimensions + 1

        super().__init__()

        self.pooling_indices = None  # created during build

    def call(self, inputs):
        inputs = self.pool_group(inputs)
        outputs = self.pooling(inputs)
        return self.restore_group_axis(outputs)

    def pool_group(self, inputs):
        if self.pool_type == 'max':
            return tf.reduce_max(inputs, axis=self.group_axis)
        else:
            return tf.reduce_mean(inputs, axis=self.group_axis)

    def restore_group_axis(self, outputs):
        if self.pooling.keepdims:
            return tf.expand_dims(outputs, axis=self.group_axis)
        else:
            return outputs

    def build(self, input_shape):
        reshaped_input = self.group_transforms.build(input_shape)
        self.pooling.build(reshaped_input)
        self.pooling_indices = self.group_transforms.compute_pooling_indices()

    def get_config(self):
        config = self.pooling.get_config()
        return config


class GroupMaxPooling1D(GroupPooling):
    """
    Layer for pooling on signals on a group. Allows for pooling on the grid as usual,
    but also on subgroups of the point group.
    Checks and restores equivariance in the same way as GroupConv2D.

    Additional arguments to MaxPooling1D:
    group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups.
    padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
    to maintain equivariance.
    allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
    non-equivariant padding.
    subgroup: defaults to '' meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', dimensions=1, **kwargs)


class GroupAveragePooling1D(GroupPooling):
    """
    Layer for pooling on signals on a group. Allows for pooling on the grid as usual,
    but also on subgroups of the point group.
    Checks and restores equivariance in the same way as GroupConv2D.

    Additional arguments to AveragePooling1D:
    group: one of the wallpaper groups, string name or the object itself, see groups.group_dict for implemented groups.
    padding: takes two additional values `valid_equiv` and `same_equiv`, that pad the minimal extra amount
    to maintain equivariance.
    allow_non_equivariance: set to true along with setting padding to either `valid` or `same` to insist using
    non-equivariant padding.
    subgroup: defaults to '' meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', dimensions=1, **kwargs)


class GroupMaxPooling2D(GroupPooling):
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
    subgroup: defaults to '' meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', dimensions=2, **kwargs)


class GroupAveragePooling2D(GroupPooling):
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
    subgroup: defaults to '' meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', dimensions=2, **kwargs)


class GroupMaxPooling3D(GroupPooling):
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
    subgroup: defaults to '' meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', dimensions=3, **kwargs)


class GroupAveragePooling3D(GroupPooling):
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
    subgroup: defaults to '' meaning no pooling over the point group. Can be set to the name of a subgroup,
    which will pool over the cosets of that subgroup, maintaining equivariance only to the subgroup.
    """
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', dimensions=3, **kwargs)


class GlobalGroupMaxPooling1D(GlobalGroupPooling):
    """As GlobalMaxPooling1D but for signals on group, pooling over the whole group."""
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', dimensions=1, **kwargs)


class GlobalGroupAveragePooling1D(GlobalGroupPooling):
    """As GlobalAveragePooling1D but for signals on group, pooling over the whole group."""
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', dimensions=1, **kwargs)


class GlobalGroupMaxPooling2D(GlobalGroupPooling):
    """As GlobalMaxPooling2D but for signals on group, pooling over the whole group."""
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', dimensions=2, **kwargs)


class GlobalGroupAveragePooling2D(GlobalGroupPooling):
    """As GlobalAveragePooling2D but for signals on group, pooling over the whole group."""
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', dimensions=2, **kwargs)


class GlobalGroupMaxPooling3D(GlobalGroupPooling):
    """As GlobalMaxPooling3D but for signals on group, pooling over the whole group."""
    def __init__(self, **kwargs):
        super().__init__(pool_type='max', dimensions=3, **kwargs)


class GlobalGroupAveragePooling3D(GlobalGroupPooling):
    """As GlobalAveragePooling3D but for signals on group, pooling over the whole group."""
    def __init__(self, **kwargs):
        super().__init__(pool_type='average', dimensions=3, **kwargs)
