from keras import ops
import tensorflow as tf


class Group:
    """
    Class representing a group along with its actions on a grid.

    Args:
        name: string, name of the group.
        order: int, the number of elements in the group.
        inverses: 1d tensor indicating the inverse of element i at position i.
        composition: 2d tensor with entry composition[r][c] the index of the group element
            obtained by composing the inverse of the rth element with the cth element.
        subgroup: dictionary from subgroup strings to indices representing the elements in
            the subgroup.
        cosets: dictionary from subgroup strings to indices representing the elementary coset
            representatives of the corresponding subgroup.
        action: callable, the action of the group on a signal on the grid.
        parent: Group, the parent group of the group.

    Methods:
        action: performs the action of the whole group on a signal on the grid
            or on the group itself.
    """

    def __init__(
        self,
        name: str,
        order: int,
        inverses=None,
        composition=None,
        subgroup=None,
        cosets=None,
        action=None,
        parent=None,
    ):
        self.name = name
        self.order = order
        self.parent = parent
        self.composition = self._compute_composition(composition)
        self.inverses = self._compute_inverses(inverses)
        self.subgroup = self._compute_subgroup(subgroup)
        self.cosets = self._compute_cosets(cosets)
        self._action = self._compute_action(action)

    def action(
        self,
        signal,
        spatial_axes: tuple = (1, 2),
        new_group_axis: int = 0,
        group_axis=None,
        acting_group: str = "",
        domain_group: str = "",
    ):
        """
        The action of the group on a given signal.

        Args:
            signal: The tensor to act on.
            spatial_axes: Tuple indicating which are the spatial axes, defaults to (1, 2).
            new_group_axis: Which axis in the output to concatenate the group elements' actions
                on, defaults to 0.
            group_axis: The group axis of the input, defaults to None, meaning a signal only
                on the grid.
            acting_group: Name of subgroup with which to act, defaults to None meaning
                the whole group.
            domain_group: Name of subgroup signal lives on, defaults to None meaning
                the whole group.

        Returns:
            Tensor of the signal acted on by the group.
        """
        acting_group, domain_group = self.parse_subgroups(acting_group, domain_group)
        kwargs = {
            "signal": signal,
            "new_group_axis": new_group_axis,
            "spatial_axes": spatial_axes,
            "acting_group": acting_group,
        }
        if domain_group is None:
            return self._action_on_grid(**kwargs)
        else:
            return self._action_on_group(group_axis=group_axis, domain_group=domain_group, **kwargs)

    def _action_on_grid(self, signal, new_group_axis: int, spatial_axes: tuple, acting_group: str):
        transformed_signal = self._action(
            signal, spatial_axes=spatial_axes, new_group_axis=new_group_axis
        )
        transformed_signal = tf.gather(
            transformed_signal, axis=new_group_axis, indices=self.subgroup[acting_group]
        )
        return transformed_signal

    def _action_on_group(
        self,
        signal,
        group_axis: int,
        new_group_axis: int,
        spatial_axes: tuple,
        acting_group: str,
        domain_group: str,
    ):
        """
        Act on a signal on the group.

        If acting_group is set to a subgroup, only act with that subgroup on a signal on
        the whole group.
        If domain_group is set to a subgroup, act with whole group on a signal on the subgroup,
        where the signal is set to 0 outside of the subgroup.
        """
        acting_group_order = len(self.subgroup[acting_group])
        domain_group_order = len(self.subgroup[domain_group])
        # fill in zeroes outside of domain group if necessary
        if domain_group_order < self.order:
            signal = self.upsample(signal, group_axis, domain_group)

        # action on grid
        transformed_signal = self._action_on_grid(
            signal, new_group_axis=group_axis, spatial_axes=spatial_axes, acting_group=acting_group
        )

        # act on point group
        shape = transformed_signal.shape
        transformed_signal = tf.reshape(
            transformed_signal,
            shape[:group_axis] + (acting_group_order * self.order) + shape[group_axis + 2 :],
        )
        composition_indices = self._composition_flat_indices(acting_group, domain_group)
        transformed_signal = tf.gather(
            transformed_signal, axis=group_axis, indices=composition_indices
        )
        transformed_signal = tf.reshape(
            transformed_signal,
            shape[:group_axis] + (acting_group_order, domain_group_order) + shape[group_axis + 2 :],
        )

        # put the acting group as the specified axis, keeping the order of the other axes the same
        permuted_axes = list(range(transformed_signal.shape.rank))
        permuted_axes = permuted_axes[:group_axis] + permuted_axes[group_axis + 1 :]
        permuted_axes = (
            permuted_axes[:new_group_axis] + [group_axis] + permuted_axes[new_group_axis:]
        )
        transformed_signal = ops.transpose(transformed_signal, permuted_axes)

        return transformed_signal

    def upsample(self, signal, group_axis, domain_group):
        domain_group_indices = self.subgroup[domain_group]
        zeros = tf.zeros(
            shape=signal.shape[:group_axis] + (1,) + signal.shape[group_axis + 1 :],
            dtype=signal.dtype,
        )
        filled_signal = []
        index_to_subgroup = {val: index for index, val in enumerate(domain_group_indices)}
        for i in range(self.order):
            if i in domain_group_indices:
                filled_signal.append(
                    tf.gather(
                        signal,
                        axis=group_axis,
                        indices=[
                            index_to_subgroup[i],
                        ],
                    )
                )
            else:
                filled_signal.append(zeros)
        return ops.concatenate(filled_signal, axis=group_axis)

    def _composition_flat_indices(self, acting_group, domain_group):
        acting_inv_indices = [self.inverses[i] for i in self.subgroup[acting_group]]
        subgroup_composition = tf.gather(self.composition, axis=0, indices=acting_inv_indices)
        subgroup_composition = tf.gather(
            subgroup_composition, axis=1, indices=self.subgroup[domain_group]
        )
        group_composition_indices = tf.constant(
            [
                [i * self.order + c for c in row]
                for i, row in enumerate(subgroup_composition.numpy())
            ]
        )
        return tf.reshape(group_composition_indices, [-1])

    def _compute_inverses(self, inverses):
        if inverses is not None:
            return tf.constant(inverses)
        return [
            [c for c in range(self.order) if self.composition[r][c] == 0][0]
            for r in range(self.order)
        ]

    def _compute_composition(self, composition):
        """Compute the composition induced by the parent group."""
        if composition is not None:
            return tf.constant(composition)

        parent_indices = self.parent.subgroup[self.name]
        composition = tf.gather(self.parent.composition, indices=parent_indices, axis=0)
        composition = tf.gather(composition, indices=parent_indices, axis=1)
        composition = [
            self._convert_parent_indices(composition[r].numpy()) for r in range(self.order)
        ]
        return tf.constant(composition)

    def _compute_action(self, action):
        if action is not None:
            return action

        def subgroup_action(signal, spatial_axes, new_group_axis):
            group_transformed = self.parent._action(signal, spatial_axes, new_group_axis)
            subgroup_transformed = tf.gather(
                group_transformed, indices=self.parent.subgroup[self.name], axis=new_group_axis
            )
            return subgroup_transformed

        return subgroup_action

    def _compute_subgroup(self, subgroup):
        if subgroup is not None:
            return subgroup
        parent_subgroups = self.parent.subgroup
        this_subgroup = set(parent_subgroups[self.name])
        subgroups = {
            name: self._convert_parent_indices(indices)
            for name, indices in parent_subgroups.items()
            if set(indices).issubset(this_subgroup)
        }
        return subgroups

    def _compute_cosets(self, cosets):
        if cosets is not None:
            return cosets
        parent_cosets = self.parent.cosets
        this_subgroup = set(self.parent.subgroup[self.name])
        cosets = {
            name: self._convert_parent_indices([i for i in indices if i in this_subgroup])
            for name, indices in parent_cosets.items()
            if name in self.subgroup.keys()
        }
        return cosets

    def _convert_parent_indices(self, parent_indices):
        mapping = {
            parent: this
            for parent, this in zip(list(self.parent.subgroup[self.name]), list(range(self.order)))
        }
        return [mapping[i] for i in parent_indices]

    def parse_subgroup(self, subgroup):
        parser = {None: None, "": self.name}
        return subgroup if subgroup else parser[subgroup]

    def parse_subgroups(self, *subgroups):
        parser = {None: None, "": self.name}
        return tuple(subgroup if subgroup else parser[subgroup] for subgroup in subgroups)
