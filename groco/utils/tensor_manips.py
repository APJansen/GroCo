import tensorflow as tf


def move_axis_to_left_of(tensor, moved_axis: int, target_axis: int):
    """Puts the moved_axis to the left of the target_axis, leaving the order of the other axes invariant."""
    axes = tuple(range(tensor.shape.rank))
    axes = axes[:moved_axis] + axes[moved_axis + 1:]
    if moved_axis < target_axis:
        target_axis -= 1
    axes = axes[:target_axis] + (moved_axis,) + axes[target_axis:]
    return tf.transpose(tensor, axes)


def split_axes(tensor, factor: int, split_axis: int, target_axis: int):
    """
    Split split_axis, dividing its size by factor, putting the new axis directly to its left,
    which is then moved to the target_axis index.
    """
    shape = list(tensor.shape)
    shape[split_axis] //= factor
    shape = tf.TensorShape(shape[:split_axis] + [factor] + shape[split_axis:])
    shape = [s if s is not None else -1 for s in shape]
    tensor_reshaped = tf.reshape(tensor, shape)
    tensor_transposed = move_axis_to_left_of(tensor_reshaped, moved_axis=split_axis, target_axis=target_axis)
    return tensor_transposed


def get_index_tensor(tensor):
    """
    Return a tensor of indices that reproduces the input through
    `tf.gather(tf.reshape(tensor, -1), indices=indices)`
    """
    if isinstance(tensor, tf.TensorShape):
        return tf.reshape(tf.range(tf.reduce_prod(tensor)), tensor)
    else:
        return tf.reshape(tf.range(tf.size(tensor)), tensor.shape)


def merge_axes(tensor, merged_axis: int, target_axis: int):
    """Transpose the merge_axis to the left of the target_axis and then merge them."""
    shape = merge_shapes(tensor.shape, merged_axis=merged_axis, target_axis=target_axis)
    shape = [s if s is not None else -1 for s in shape]
    transposed_tensor = move_axis_to_left_of(tensor, moved_axis=merged_axis, target_axis=target_axis)
    return tf.reshape(transposed_tensor, shape)


def merge_shapes(shape, merged_axis: int, target_axis: int):
    shape = list(shape)
    shape[target_axis] *= shape[merged_axis]
    shape.pop(merged_axis)
    return shape
