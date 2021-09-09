import tensorflow as tf


class Group:
    """
    Class representing the point group  of a wallpaper group.

    Attributes:
        order: the number of elements.
        inverses: 1d tensor indicating the inverse of element i at position i.
        composition: 2d tensor with entry composition[r][c] the index of the group element obtained by composing
        the inverse of the rth element with the cth element
        subgroup: dictionary from subgroup strings to indices representing the elements in the subgroup.
        cosets: dictionary from subgroup strings to indices representing the elementary coset representatives of the
        corresponding subgroup
        name: string name of the group.

    Methods:
        action: performs the action of the whole group on a signal on the grid or on the group itself.
    """
    def __init__(self,
                 order: int,
                 inverses: list,
                 composition: list,
                 subgroup: dict,
                 cosets: dict,
                 action,
                 name: str):
        self.order = order
        self.inverses = tf.constant(inverses)
        self.composition = tf.constant(composition)
        self.subgroup = subgroup
        self.cosets = cosets
        self._action = action
        self.name = name

    def action(self, signal, spatial_axes: tuple = (1, 2), new_group_axis: int = 0, group_axis=None, subgroup: str = None):
        """
        The action of the group on a given signal.

        :param signal: The tensor to act on.
        :param spatial_axes: Tuple indicating which are the spatial axes, defaults to (1, 2).
        :param new_group_axis: Which axis in the output to concatenate the group elements' actions on, defaults to 0.
        :param group_axis: The group axis of the input, defaults to None, meaning a signal only on the grid.
        :param subgroup: Name of subgroup with which to act, defaults to None meaning the whole group.
        :return: Tensor of the signal acted on by the group.
        """
        if group_axis is None:
            return self._action_on_grid(signal, new_group_axis=new_group_axis, spatial_axes=spatial_axes,
                                        subgroup=subgroup)
        else:
            return self._action_on_group(signal,
                                         spatial_axes=spatial_axes,
                                         group_axis=group_axis,
                                         new_group_axis=new_group_axis,
                                         subgroup=subgroup)

    def _action_on_grid(self, signal, new_group_axis: int, spatial_axes: tuple, subgroup: str = None):
        subgroup_name = self.name if subgroup is None else subgroup
        subgroup_indices = self.subgroup[subgroup_name]

        transformed_signal = self._action(signal, spatial_axes=spatial_axes, new_group_axis=new_group_axis)
        transformed_signal = tf.gather(transformed_signal, axis=new_group_axis, indices=subgroup_indices)

        return transformed_signal

    def _action_on_group(self, signal, group_axis: int, new_group_axis: int, spatial_axes: tuple,
                         subgroup: str = None):
        """
        Act on a signal on the group, potentially only with a subgroup.
        """
        # action on grid
        transformed_signal = self._action_on_grid(signal, new_group_axis=group_axis, spatial_axes=spatial_axes)

        subgroup_name = self.name if subgroup is None else subgroup
        subgroup_indices = self.subgroup[subgroup_name]
        transformed_signal = tf.gather(transformed_signal, axis=group_axis, indices=self.subgroup[subgroup_name])

        # act on point group
        subgroup_order = len(subgroup_indices)
        shape = transformed_signal.shape
        transformed_signal = tf.reshape(transformed_signal,
                                        shape[:group_axis] + (subgroup_order * self.order) + shape[group_axis + 2:])

        composition_indices = self._composition_flat_indices(subgroup_name)

        transformed_signal = tf.gather(transformed_signal, axis=group_axis, indices=composition_indices)
        transformed_signal = tf.reshape(transformed_signal, shape)

        # put the acting group as the specified axis, keeping the order of the other axes the same
        permuted_axes = list(range(transformed_signal.shape.rank))
        permuted_axes = permuted_axes[:group_axis] + permuted_axes[group_axis + 1:]
        permuted_axes = permuted_axes[:new_group_axis] + [group_axis] + permuted_axes[new_group_axis:]
        transformed_signal = tf.transpose(transformed_signal, permuted_axes)

        return transformed_signal

    def _composition_flat_indices(self, subgroup_name):
        subgroup_indices = self.subgroup[subgroup_name]

        subgroup_composition = tf.gather(self.composition, axis=0, indices=subgroup_indices)
        group_composition_indices = tf.constant([[i * self.order + c for c in row] for i, row in
                                                 enumerate(subgroup_composition.numpy())])
        return tf.reshape(group_composition_indices, [-1])
