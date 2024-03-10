from keras import KerasTensor, ops


def get_index_tensor(tensor) -> KerasTensor:
    """
    Return a tensor of indices `index_tensor` that, when given an `input_tensor` of the same
    shape as the input `tensor` to this function, satisfies the identity:

        `input_tensor == ops.take(ops.reshape(input_tensor, -1), indices=index_tensor)`

    Args:
        tensor: KerasTensor.

    Returns:
        A KerasTensor of indices.
    """
    shape = tensor.shape
    num_elements = ops.prod(shape)
    index_tensor = ops.reshape(ops.arange(num_elements), shape)
    return index_tensor


def move_axis_to_left_of(tensor: KerasTensor, moved_axis: int, target_axis: int) -> KerasTensor:
    """
    Put the `moved_axis` to the left of the `target_axis`, leaving the order of the other
    axes invariant.

    Args:
        tensor: The tensor to have its axes moved.
        moved_axis: The index of the axis to be moved.
        target_axis: The index of the axis to the left of which the moved_axis is placed.

    Returns:
        The tensor with its axes moved.
    """
    axes = tuple(range(tensor.shape.rank))
    axes = axes[:moved_axis] + axes[moved_axis + 1 :]
    if (
        moved_axis < target_axis
    ):  # in this case after removing moved_axis, target_axis shifts left by one
        target_axis -= 1
    axes = axes[:target_axis] + (moved_axis,) + axes[target_axis:]
    return ops.transpose(tensor, axes)


def split_axes(tensor: KerasTensor, factor: int, split_axis: int, target_axis: int) -> KerasTensor:
    """
    Split `split_axis`, dividing its size by factor, putting the new axis directly to its left,
    which is then moved to the `target_axis` index.

    Args:
        tensor: The tensor to have its axis split.
        factor: The number of elements to split from the split_axis.
        split_axis: The index of the axis to be split.
        target_axis: The index of the new axis.

    Returns:
        The tensor with its axis split.
    """
    new_shape = split_shapes(tensor.shape, factor, split_axis)
    new_shape = format_batch_dim(new_shape)
    tensor_reshaped = ops.reshape(tensor, new_shape)
    tensor_transposed = move_axis_to_left_of(
        tensor_reshaped, moved_axis=split_axis, target_axis=target_axis
    )
    return tensor_transposed


def split_shapes(shape, factor: int, split_axis: int) -> list:
    """
    Split one axis of a shape into two, moving `factor` elements to the left of the
    `split_axis` and dividing the `split_axis` itself by the same `factor`.

    Args:
        shape: The full shape.TensorShape.
        factor: The number of elements to split off.
        split_axis: The index of the axis to be split.

    Returns:
        A list representing the split shape.
    """
    shape = list(shape)
    shape[split_axis] //= factor
    shape = shape[:split_axis] + [factor] + shape[split_axis:]
    return shape


def merge_axes(tensor: KerasTensor, merged_axis: int, target_axis: int) -> KerasTensor:
    """
    Transpose the `merged_axis` to the left of the `target_axis` and then merge them.

    Args:
        tensor: The tensor to have its axes merged.
        merged_axis: The index of the axis to be removed.
        target_axis: The index of the axis to be enlarged.

    Returns:
        The tensor with its axes merged.
    """
    shape = merge_shapes(tensor.shape, merged_axis=merged_axis, target_axis=target_axis)
    shape = format_batch_dim(shape)
    transposed_tensor = move_axis_to_left_of(
        tensor, moved_axis=merged_axis, target_axis=target_axis
    )
    return ops.reshape(transposed_tensor, shape)


def merge_shapes(shape, merged_axis: int, target_axis: int) -> list:
    """
    Merge two axes of a shape into one, removing `merged_axis` and multiplying the size
    of the `target_axis` by the size of the `merged_axis`.

    Args:
        shape: The full shape.
        merged_axis: The index of the axis to remove.
        target_axis: The index of the axis to add to.

    Returns:
        A list representing the merged shape.
    """
    shape = list(shape)
    shape[target_axis] *= shape[merged_axis]
    shape.pop(merged_axis)
    return shape


def format_batch_dim(shape: list) -> list:
    """
    Turn any None entry in a shape, which represents an unknown batch size, by a -1,
    which represents all remaining elements.
    Need to do this because None is not supported by ops.reshape.

    Args:
        shape: A list representing a shape with potentially a None for the batch dimension.

    Returns:
        A list with any None replaced by -1.
    """
    return [s if s is not None else -1 for s in shape]
