import tensorflow as tf
from functools import partial


class WallpaperGroup:
    def __init__(self, order: int, inverses: list, composition: list, subgroup: dict, action, name: str,
                 height_axis=0, width_axis=1, new_group_axis=2):
        self.order = order
        self.inverses = tf.constant(inverses)
        self.composition = tf.constant(composition)
        self.subgroup = subgroup
        self.action = partial(action, height_axis=height_axis, width_axis=width_axis, new_group_axis=new_group_axis)
        self.name = name

        self.composition_indices = self.compute_indices()

    def compute_indices(self):
        flattened_composition = tf.constant([[i * self.order + c for c in row] for i, row in
                                         enumerate(self.composition.numpy())])
        flattened_composition = tf.reshape(flattened_composition, (self.order * self.order))
        return flattened_composition


def P4M_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    kernel = tf.concat([kernel, tf.reverse(kernel, axis=[width_axis])], axis=new_group_axis)
    kernel = tf.concat([kernel, tf.reverse(kernel, axis=[height_axis])], axis=new_group_axis)
    axes = list(range(kernel.shape.rank))
    axes[height_axis], axes[width_axis] = axes[width_axis], axes[height_axis]
    kernel = tf.concat([kernel, tf.transpose(kernel, axes)], axis=new_group_axis)

    # this line is to make the order (e, R, R^2, R^3, F, R F, R^2 F, R^3 F)
    kernel = tf.gather(kernel, axis=new_group_axis, indices=(0, 5, 3, 6, 1, 4, 2, 7))

    return kernel

def P4_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    kernel = tf.concat([kernel, tf.reverse(kernel, axis=[width_axis, height_axis])], axis=new_group_axis)
    axes = list(range(kernel.shape.rank))
    axes[height_axis], axes[width_axis] = axes[width_axis], axes[height_axis]
    kernel = tf.concat([kernel, tf.reverse(tf.transpose(kernel, axes), axis=[height_axis])], axis=new_group_axis)
    kernel = tf.gather(kernel, axis=new_group_axis, indices=[0, 2, 1, 3])

    return kernel

def P2MM_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    kernel = tf.concat([kernel, tf.reverse(kernel, axis=[width_axis])], axis=new_group_axis)
    kernel = tf.concat([kernel, tf.reverse(kernel, axis=[height_axis])], axis=new_group_axis)

    return kernel

def PMh_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    flipped_kernel = tf.reverse(kernel, axis=[height_axis])
    kernel = tf.concat([kernel, flipped_kernel], axis=new_group_axis)
    return kernel

def PMw_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    flipped_kernel = tf.reverse(kernel, axis=[width_axis])
    kernel = tf.concat([kernel, flipped_kernel], axis=new_group_axis)
    return kernel

def P2_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    rotated_kernel = tf.reverse(kernel, axis=[width_axis, height_axis])
    kernel = tf.concat([kernel, rotated_kernel], axis=new_group_axis)
    return kernel

def P1_action(kernel, height_axis, width_axis, new_group_axis):
    kernel = tf.expand_dims(kernel, axis=new_group_axis)
    return kernel


P4M = WallpaperGroup(
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
    action=P4M_action
)

P4 = WallpaperGroup(
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
    action=P4_action
)

P2MM = WallpaperGroup(
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
    action=P2MM_action
)

PMh = WallpaperGroup(
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
    action=PMh_action
)

PMw = WallpaperGroup(
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
    action=PMw_action
)

P2 = WallpaperGroup(
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
    action=P2_action
)

P1 = WallpaperGroup(
    name='P1',
    order=1,
    inverses=[0],
    composition=[[0]],
    subgroup={'P1': [0]},
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