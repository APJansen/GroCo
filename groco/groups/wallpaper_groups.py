from keras import ops

from geqco.groups.group import Group


def P4M_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = ops.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    signal = ops.concatenate([signal, ops.flip(signal, axis=width_axis)], axis=new_group_axis)
    signal = ops.concatenate([signal, ops.flip(signal, axis=height_axis)], axis=new_group_axis)
    axes = list(range(ops.ndim(signal)))
    axes[height_axis], axes[width_axis] = axes[width_axis], axes[height_axis]
    signal = ops.concatenate([signal, ops.transpose(signal, axes)], axis=new_group_axis)
    # this line is to make the order (e, R, R^2, R^3, F, R F, R^2 F, R^3 F)
    signal = ops.take(signal, axis=new_group_axis, indices=(0, 5, 3, 6, 1, 4, 2, 7))

    return signal


def P4_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = ops.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    signal = ops.concatenate(
        [signal, ops.flip(ops.flip(signal, axis=width_axis), axis=height_axis)], axis=new_group_axis
    )
    axes = list(range(ops.ndim(signal)))
    axes[height_axis], axes[width_axis] = axes[width_axis], axes[height_axis]
    signal = ops.concatenate(
        [signal, ops.flip(ops.transpose(signal, axes), axis=height_axis)], axis=new_group_axis
    )
    signal = ops.take(signal, axis=new_group_axis, indices=[0, 2, 1, 3])
    return signal


def P2MM_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = ops.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    signal = ops.concatenate([signal, ops.flip(signal, axis=width_axis)], axis=new_group_axis)
    signal = ops.concatenate([signal, ops.flip(signal, axis=height_axis)], axis=new_group_axis)

    return signal


def PMh_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = ops.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    flipped_kernel = ops.flip(signal, axis=height_axis)
    signal = ops.concatenate([signal, flipped_kernel], axis=new_group_axis)
    return signal


def PMw_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = ops.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    flipped_kernel = ops.flip(signal, axis=width_axis)
    signal = ops.concatenate([signal, flipped_kernel], axis=new_group_axis)
    return signal


def P2_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    height_axis, width_axis = spatial_axes
    signal = ops.expand_dims(signal, axis=new_group_axis)
    if new_group_axis <= height_axis:
        height_axis += 1
    if new_group_axis <= width_axis:
        width_axis += 1

    rotated_kernel = ops.flip(ops.flip(signal, axis=width_axis), axis=height_axis)
    signal = ops.concatenate([signal, rotated_kernel], axis=new_group_axis)
    return signal


def P1_action(signal, spatial_axes=(0, 1), new_group_axis=2):
    signal = ops.expand_dims(signal, axis=new_group_axis)
    return signal


P4M = Group(
    name="P4M",
    order=8,
    inverses=[0, 3, 2, 1, 4, 5, 6, 7],
    composition=[
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 0, 1, 2, 7, 4, 5, 6],
        [4, 7, 6, 5, 0, 3, 2, 1],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 5, 4, 7, 2, 1, 0, 3],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ],
    subgroup={
        "P4M": [0, 1, 2, 3, 4, 5, 6, 7],
        "P4": [0, 1, 2, 3],
        "P2MM": [0, 4, 6, 2],
        "PMw": [0, 4],
        "PMh": [0, 6],
        "P2": [0, 2],
        "P1": [0],
    },
    cosets={
        "P4M": [0],
        "P4": [0, 4],
        "P2MM": [0, 1],
        "PMw": [0, 1, 2, 3],
        "PMh": [0, 1, 2, 3],
        "P2": [0, 1, 4, 5],
        "P1": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    action=P4M_action,
)
P4 = Group(
    name="P4",
    order=4,
    inverses=[0, 3, 2, 1],
    composition=[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]],
    subgroup={"P4": [0, 1, 2, 3], "P2": [0, 2], "P1": [0]},
    cosets={"P4": [0], "P2": [0, 1], "P1": [0, 1, 2, 3]},
    action=P4_action,
)
P2MM = Group(
    name="P2MM",
    order=4,
    inverses=[0, 1, 2, 3],
    composition=[[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
    subgroup={"P2MM": [0, 1, 2, 3], "PMh": [0, 2], "PMw": [0, 1], "P1": [0]},
    cosets={"P2MM": [0], "PMh": [0, 1], "PMw": [0, 2], "P1": [0, 1, 2, 3]},
    action=P2MM_action,
)
PMh = Group(
    name="PMh",
    order=2,
    inverses=[0, 1],
    composition=[[0, 1], [1, 0]],
    subgroup={"PMh": [0, 1], "P1": [0]},
    cosets={"PMh": [0], "P1": [0, 1]},
    action=PMh_action,
)
PMw = Group(
    name="PMw",
    order=2,
    inverses=[0, 1],
    composition=[[0, 1], [1, 0]],
    subgroup={"PMw": [0, 1], "P1": [0]},
    cosets={"PMw": [0], "P1": [0, 1]},
    action=PMw_action,
)
P2 = Group(
    name="P2",
    order=2,
    inverses=[0, 1],
    composition=[[0, 1], [1, 0]],
    subgroup={"P2": [0, 1], "P1": [0]},
    cosets={"P2": [0], "P1": [0, 1]},
    action=P2_action,
)
P1 = Group(
    name="P1",
    order=1,
    inverses=[0],
    composition=[[0]],
    subgroup={"P1": [0]},
    cosets={"P1": [0]},
    action=P1_action,
)
wallpaper_group_dict = {
    "P1": P1,
    "P2": P2,
    "PMw": PMw,
    "PMh": PMh,
    "P2MM": P2MM,
    "P4": P4,
    "P4M": P4M,
}
