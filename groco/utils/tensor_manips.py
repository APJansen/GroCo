import tensorflow as tf


def get_index_tensor(tensor) -> tf.Tensor:
    """
    Return a tensor of indices `index_tensor` that, when given an `input_tensor` of the same shape as the input
    `tensor` to this function, satisfies the identity:

        `input_tensor == tf.gather(tf.reshape(input_tensor, -1), indices=index_tensor)`

    :param tensor: Either a tf.Tensor or a tf.TensorShape.
    :return: A tf.Tensor of indices.
    """
    shape = tensor if isinstance(tensor, tf.TensorShape) else tensor.shape
    num_elements = tf.reduce_prod(shape)
    index_tensor = tf.reshape(tf.range(num_elements), shape)
    return index_tensor


def move_axis_to_left_of(tensor: tf.Tensor, moved_axis: int, target_axis: int) -> tf.Tensor:
    """
    Put the `moved_axis` to the left of the `target_axis`, leaving the order of the other axes invariant.

    :param tensor: The tensor to have its axes moved.
    :param moved_axis: The index of the axis to be moved.
    :param target_axis: The index of the axis to the left of which the moved_axis is placed.
    :return: The tensor with its axes moved.
    """
    axes = tuple(range(tensor.shape.rank))
    axes = axes[:moved_axis] + axes[moved_axis + 1 :]
    if (
        moved_axis < target_axis
    ):  # in this case after removing moved_axis, target_axis shifts left by one
        target_axis -= 1
    axes = axes[:target_axis] + (moved_axis,) + axes[target_axis:]
    return tf.transpose(tensor, axes)


def split_axes(tensor: tf.Tensor, factor: int, split_axis: int, target_axis: int) -> tf.Tensor:
    """
    Split `split_axis`, dividing its size by factor, putting the new axis directly to its left,
    which is then moved to the `target_axis` index.

    :param tensor: The tensor to have its axis split.
    :param factor: The number of elements to split from the split_axis.
    :param split_axis: The index of the axis to be split.
    :param target_axis: The index of the new axis.
    :return: The tensor with its axis split.
    """
    new_shape = split_shapes(tensor.shape, factor, split_axis)
    new_shape = format_batch_dim(new_shape)
    tensor_reshaped = tf.reshape(tensor, new_shape)
    tensor_transposed = move_axis_to_left_of(
        tensor_reshaped, moved_axis=split_axis, target_axis=target_axis
    )
    return tensor_transposed


def split_shapes(shape, factor: int, split_axis: int) -> list:
    """
    Split one axis of a shape into two, moving `factor` elements to the left of the `split_axis` and dividing
    the `split_axis` itself by the same `factor`.

    :param shape: The full shape, tuple or tf.TensorShape.
    :param factor: The number of elements to split off.
    :param split_axis: The index of the axis to be split.
    :return: A list representing the split shape.
    """
    shape = list(shape)
    shape[split_axis] //= factor
    shape = tf.TensorShape(shape[:split_axis] + [factor] + shape[split_axis:])
    return shape


def merge_axes(tensor: tf.Tensor, merged_axis: int, target_axis: int) -> tf.Tensor:
    """
    Transpose the `merged_axis` to the left of the `target_axis` and then merge them.

    :param tensor: The tensor to have its axes merged.
    :param merged_axis: The index of the axis to be removed.
    :param target_axis: The index of the axis to be enlarged.
    :return: The tensor with its axes merged.
    """
    shape = merge_shapes(tensor.shape, merged_axis=merged_axis, target_axis=target_axis)
    shape = format_batch_dim(shape)
    transposed_tensor = move_axis_to_left_of(
        tensor, moved_axis=merged_axis, target_axis=target_axis
    )
    return tf.reshape(transposed_tensor, shape)


def merge_shapes(shape, merged_axis: int, target_axis: int) -> list:
    """
    Merge two axes of a shape into one, removing `merged_axis` and multiplying the size of the `target_axis` by the size
    of the `merged_axis`.

    :param shape: The full shape, tuple or tf.TensorShape.
    :param merged_axis: The index of the axis to remove.
    :param target_axis: The index of the axis to add to.
    :return: A list representing the merged shape.
    """
    shape = list(shape)
    shape[target_axis] *= shape[merged_axis]
    shape.pop(merged_axis)
    return shape


def format_batch_dim(shape: list) -> list:
    """
    Turn any None entry in a shape, which represents an unknown batch size, by a -1, which represents all
    remaining elements. Need to do this because None is not supported by tf.reshape.

    :param shape: A list representing a shape with potentially a None for the batch dimension.
    :return: A list with any None replaced by -1.
    """
    return [s if s is not None else -1 for s in shape]
