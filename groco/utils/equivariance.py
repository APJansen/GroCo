from keras import ops
import tensorflow as tf


def check_equivariance(
    layer,
    signal,
    group=None,
    spatial_axes: tuple = (1, 2),
    group_axis=None,
    acting_group="",
    domain_group="",
    target_group="",
):
    """
    Test the equivariance of a `layer` L under the transformation of an `acting_group` G on a `signal` s, by computing max |GL(s) - LG(s)|.

    The layer L takes signal on the `domain_group` to signals on the `target_group`
    All of `acting_group`, `domain_group` and `target_group` should be strings representing subgroups of `group`.
    `group` itself will be read off from the layer if not set.

    Args:
        layer: The layer to test, any callable, but it must treat the first dimension as a
            batch dimension and conform to the input and output groups specified.
        signal: The input signal on `domain_group` to test on.
        group: The group, defaults to None in which case layer.group will be used.
        spatial_axes: tuple of integers indicating the spatial axes.
        group_axis: axis indexing group elements, the layer should keep them in this axis.
        acting_group: defaults to '', in which case the full group will be used.
        domain_group: defaults to '', in which case the full group will be used.
        target_group: defaults to '', in which case the full group will be used.

    Returns:
        maximal absolute difference between first transforming vs first applying layer.
    """
    group = layer.group if group is None else group
    acting_group, domain_group, target_group = group.parse_subgroups(
        acting_group, domain_group, target_group
    )
    _check_shape(signal, group_axis, group, domain_group, "signal", "domain group")

    layer_signal = layer(signal)
    _check_shape(
        layer_signal,
        group_axis,
        group,
        target_group,
        "layer(signal)",
        f"target group {target_group}",
    )

    g_signal = group.action(
        signal,
        spatial_axes=spatial_axes,
        group_axis=group_axis,
        new_group_axis=0,
        acting_group=acting_group,
        domain_group=domain_group,
    )

    layer_g_signal = _act_on_batched_transform(layer, g_signal)

    g_layer_signal = group.action(
        layer_signal,
        spatial_axes=spatial_axes,
        group_axis=group_axis,
        new_group_axis=0,
        acting_group=acting_group,
        domain_group=target_group,
    )

    diffs = layer_g_signal - g_layer_signal
    maxdiff = tf.reduce_max(tf.abs(diffs))
    return maxdiff


def _check_shape(signal, group_axis, group, subgroup, signal_string, group_string):
    if subgroup is not None:
        message = (
            signal_string
            + " not on "
            + group_string
            + ", "
            + f"expected shape {len(group.subgroup[subgroup])} in axis {group_axis}, "
            f"got {signal.shape[group_axis]}."
        )
        assert signal.shape[group_axis] == len(group.subgroup[subgroup]), message


def _act_on_batched_transform(layer, g_signal):
    shape = g_signal.shape
    g_order, batch_size = shape[:2]
    signal_merged = ops.reshape(g_signal, (g_order * batch_size,) + shape[2:])
    layer_signal = layer(signal_merged)
    layer_signal = ops.reshape(layer_signal, (g_order, batch_size) + layer_signal.shape[1:])
    return layer_signal
