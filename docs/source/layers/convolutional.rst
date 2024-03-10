Convolutional Layers
====================

All the convolution layers follow the same pattern.
They support all of the functionality of the regular convolution (although not everything is tested).

They all have the additional arguments:

* `group` (str) - The group with which to apply the convolution.
* `subgroup` (str, default None) - To use a subgroup to convolve a signal on the full group.
  Defaults to None, which means that the subgroup is the same as the group.
* `allow_non_equivariance` (default: False) - whether to allow non-equivariant operations.
  This is used to implement non-equivariant convolutions.

Furthermore there are additional options for padding:

* `padding='valid_equiv'` - Valid padding with minimal extra padding to ensure equivariance.
* `padding='same_equiv'` - Same padding with minimal extra padding to ensure equivariance.


.. toctree::
   :maxdepth: 3

   conv2d
   conv3d
   conv2d_transpose
   conv3d_transpose
