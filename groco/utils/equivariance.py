import tensorflow as tf
from groco.groups import group_dict


def test_equivariance(
        layer, signal, group=None, spatial_axes: tuple = (1, 2), group_axis=None,
        acting_group='', domain_group='', target_group=''):
    """
    Test the equivariance of a layer L under the transformation of a (sub)group G on a signal s,
    by computing max |GL(s) - LG(s)|.

    :param layer: The layer to test, can be any Keras layer but if it does not have a group attribute that needs to
    be specified separately under `group_name`.
    :param signal: The input signal to test on.
    :param group:the group, defaults to None in which case layer.group will be used.
    :param spatial_axes: tuple of integers indicating the spatial axes.
    :param group_axis: defaults to None, in which case the input signal is on the grid only.
    :param acting_group: defaults to '', in which case the full group will be used
    :param domain_group: defaults to '', in which case the full group will be used
    :param target_group: defaults to '', in which case the full group will be used
    :return: maximal absolute difference between first transforming vs first applying layer.
    """
    group = layer.group if group is None else group
    acting_group, domain_group, target_group = group.parse_subgroups(acting_group, domain_group, target_group)

    g_signal = group.action(
        signal, spatial_axes=spatial_axes, group_axis=group_axis, new_group_axis=0,
        acting_group=acting_group, domain_group=domain_group)

    layer_signal = layer(signal)

    # merge with batch dimension
    g_signal = tf.reshape(g_signal, (g_signal.shape[0] * g_signal.shape[1]) + g_signal.shape[2:])
    layer_g_signal = layer(g_signal)

    # GroupConv2D always outputs signals on the group, but if the layer is a regular convolution it will still be
    # on the grid
    if layer_signal.shape.rank > 2 + len(spatial_axes):
        group_axis = spatial_axes[-1] + 1
    else:
        group_axis = None
    subgroup = group_dict[acting_group]
    g_layer_signal = subgroup.action(layer_signal, spatial_axes=spatial_axes, group_axis=group_axis, new_group_axis=0)
    g_layer_signal = tf.reshape(g_layer_signal,
                                (g_layer_signal.shape[0] * g_layer_signal.shape[1]) + g_layer_signal.shape[2:])

    diffs = layer_g_signal - g_layer_signal

    maxdiff = tf.reduce_max(tf.abs(diffs))
    return maxdiff
