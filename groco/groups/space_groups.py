import tensorflow as tf
from groco.groups.base import Group


def P4M_action_3d(signal, spatial_axes=(0, 1, 2), new_group_axis=2):
    height_axis, width_axis, depth_axis = spatial_axes
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


P4M_test_3d = Group(
    name='P4M_test_3d',
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
            'P4M_test_3d': [0, 1, 2, 3, 4, 5, 6, 7]
        },
    cosets={
        'P4M_test_3d': [0]
    },
    action=P4M_action_3d
)

group_dict = {
    'P4M_test_3d': P4M_test_3d
}
