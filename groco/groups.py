import tensorflow as tf


class Group:
    """
    Class representing the point group  of a wallpaper group.

    Also includes its action on signals on the grid (`group.action_on_grid`)
    and on signals on the wallpaper group (`group.action_on_group`)
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
        self.action = action
        self.name = name

    def action_on_grid(self, signal, new_group_axis: int, spatial_axes: tuple, subgroup_name: str = None):
        subgroup_name = self.name if subgroup_name is None else subgroup_name
        subgroup_indices = self.subgroup[subgroup_name]

        transformed_signal = self.action(signal, spatial_axes=spatial_axes, new_group_axis=new_group_axis)
        transformed_signal = tf.gather(transformed_signal, axis=new_group_axis, indices=subgroup_indices)

        return transformed_signal

    def action_on_group(self, signal, group_axis: int, new_group_axis: int, spatial_axes: tuple,
                        subgroup_name: str = None):
        """
        Act on a signal on the group, potentially only with a subgroup.
        """
        assert signal.shape[group_axis] == self.order, \
            f"group_axis={group_axis} does not have group.order {self.order} size but {signal.shape[group_axis]}."

        # action on grid
        transformed_signal = self.action_on_grid(signal, new_group_axis=group_axis, spatial_axes=spatial_axes)

        # act on point group
        subgroup_name = self.name if subgroup_name is None else subgroup_name
        subgroup_indices = self.subgroup[subgroup_name]
        subgroup_order = len(subgroup_indices)
        shape = transformed_signal.shape
        transformed_signal = tf.reshape(transformed_signal,
                                        shape[:group_axis] + (subgroup_order * self.order) + shape[group_axis + 2:])

        composition_indices = self.composition_flat_indices(subgroup_name)

        transformed_signal = tf.gather(transformed_signal, axis=group_axis, indices=composition_indices)

        transformed_signal = tf.reshape(transformed_signal, shape)

        # put the acting group as the specified axis, keeping the order of the other axes the same
        permuted_axes = list(range(transformed_signal.shape.rank))
        permuted_axes = permuted_axes[:group_axis] + permuted_axes[group_axis + 1:]
        permuted_axes = permuted_axes[:new_group_axis] + [group_axis] + permuted_axes[new_group_axis:]
        transformed_signal = tf.transpose(transformed_signal, permuted_axes)

        return transformed_signal

    def composition_flat_indices(self, subgroup_name):
        subgroup_indices = self.subgroup[subgroup_name]

        subgroup_composition = tf.gather(self.composition, axis=0, indices=subgroup_indices)
        group_composition_indices = tf.constant([[i * self.order + c for c in row] for i, row in
                                                 enumerate(subgroup_composition.numpy())])
        return tf.reshape(group_composition_indices, [-1])


def P4M_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = tf.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    signal = tf.concat([signal, tf.reverse(signal, axis=[width_axis])], axis=new_group_axis)
    signal = tf.concat([signal, tf.reverse(signal, axis=[height_axis])], axis=new_group_axis)
    axes = list(range(signal.shape.rank))
    axes[height_axis], axes[width_axis] = axes[width_axis], axes[height_axis]
    signal = tf.concat([signal, tf.transpose(signal, axes)], axis=new_group_axis)
    # this line is to make the order (e, R, R^2, R^3, F, R F, R^2 F, R^3 F)
    signal = tf.gather(signal, axis=new_group_axis, indices=(0, 5, 3, 6, 1, 4, 2, 7))

    return signal


def P4_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = tf.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    signal = tf.concat([signal, tf.reverse(signal, axis=[width_axis, height_axis])], axis=new_group_axis)
    axes = list(range(signal.shape.rank))
    axes[height_axis], axes[width_axis] = axes[width_axis], axes[height_axis]
    signal = tf.concat([signal, tf.reverse(tf.transpose(signal, axes), axis=[height_axis])], axis=new_group_axis)
    signal = tf.gather(signal, axis=new_group_axis, indices=[0, 2, 1, 3])
    return signal


def P2MM_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = tf.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    signal = tf.concat([signal, tf.reverse(signal, axis=[width_axis])], axis=new_group_axis)
    signal = tf.concat([signal, tf.reverse(signal, axis=[height_axis])], axis=new_group_axis)

    return signal


def PMh_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = tf.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    flipped_kernel = tf.reverse(signal, axis=[height_axis])
    signal = tf.concat([signal, flipped_kernel], axis=new_group_axis)
    return signal


def PMw_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = tf.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    flipped_kernel = tf.reverse(signal, axis=[width_axis])
    signal = tf.concat([signal, flipped_kernel], axis=new_group_axis)
    return signal


def P2_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = tf.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    rotated_kernel = tf.reverse(signal, axis=[width_axis, height_axis])
    signal = tf.concat([signal, rotated_kernel], axis=new_group_axis)
    return signal


def P1_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    signal = tf.expand_dims(signal, axis=new_group_axis)
    return signal


P4M = Group(
    name='P4M',
    order=8,
    inverses=[0, 3, 2, 1, 4, 5, 6, 7],
    composition=[
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [3, 0, 1, 2, 7, 4, 5, 6],
                    [2, 3, 0, 1, 6, 7, 4, 5],
                    [1, 2, 3, 0, 5, 6, 7, 4],
                    [4, 7, 6, 5, 0, 3, 2, 1],
                    [5, 4, 7, 6, 1, 0, 3, 2],
                    [6, 5, 4, 7, 2, 1, 0, 3],
                    [7, 6, 5, 4, 3, 2, 1, 0]],
    subgroup={
            'P4M': [0, 1, 2, 3, 4, 5, 6, 7],
            'P4': [0, 1, 2, 3],
            'P2MM': [0, 4, 6, 2],
            'PMw': [0, 4],
            'PMh': [0, 6],
            'P2': [0, 2],
            'P1': [0]
        },
    cosets={
        'P4M': [0],
        'P4': [0, 4],
        'P2MM': [0, 1],
        'PMw': [0, 1, 2, 3],
        'PMh': [0, 1, 2, 3],
        'P2': [0, 1, 4, 5],
        'P1': [0, 1, 2, 3, 4, 5, 6, 7]
    },
    action=P4M_action
)

P4 = Group(
    name='P4',
    order=4,
    inverses=[0, 3, 2, 1],
    composition= [
                    [0, 1, 2, 3],
                    [3, 0, 1, 2],
                    [2, 3, 0, 1],
                    [1, 2, 3, 0]],
    subgroup={
            'P4': [0, 1, 2, 3],
            'P2': [0, 2],
            'P1': [0]
        },
    cosets={
        'P4': [0],
        'P2': [0, 1],
        'P1': [0, 1, 2, 3]
    },
    action=P4_action
)

P2MM = Group(
    name='P2MM',
    order=4,
    inverses=[0, 1, 2, 3],
    composition=[
                    [0, 1, 2, 3],
                    [1, 0, 3, 2],
                    [2, 3, 0, 1],
                    [3, 2, 1, 0]],
    subgroup={
            'P2MM': [0, 1, 2, 3],
            'PMh': [0, 2],
            'PMw': [0, 1],
            'P1': [0]
        },
    cosets={
        'P2MM': [0],
        'PMh': [0, 1],
        'PMw': [0, 2],
        'P1': [0, 1, 2, 3]
    },
    action=P2MM_action
)

PMh = Group(
    name='PMh',
    order=2,
    inverses=[0, 1],
    composition=[
                    [0, 1],
                    [1, 0]],
    subgroup={
            'PMh': [0, 1],
            'P1': [0]
        },
    cosets={
        'PMh': [0],
        'P1': [0, 1]
    },
    action=PMh_action
)

PMw = Group(
    name='PMw',
    order=2,
    inverses=[0, 1],
    composition=[
                    [0, 1],
                    [1, 0]],
    subgroup={
            'PMw': [0, 1],
            'P1': [0]
        },
    cosets={
        'PMw': [0],
        'P1': [0, 1]
    },
    action=PMw_action
)

P2 = Group(
    name='P2',
    order=2,
    inverses=[0, 1],
    composition=[
        [0, 1],
        [1, 0]],
    subgroup={
        'P2': [0, 1],
        'P1': [0]
    },
    cosets={
        'P2': [0],
        'P1': [0, 1]
    },
    action=P2_action
)

P1 = Group(
    name='P1',
    order=1,
    inverses=[0],
    composition=[[0]],
    subgroup={'P1': [0]},
    cosets={'P1': [0]},
    action=P1_action
)

group_dict = {
    'P1': P1,
    'P2': P2,
    'PMw': PMw,
    'PMh': PMh,
    'P2MM': P2MM,
    'P4': P4,
    'P4M': P4M
}