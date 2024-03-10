API Documentation
=================

This part of the documentation covers all the interfaces of Groco.

.. toctree::
   :maxdepth: 2


Layers
--------

Convolutional layers:

.. autoclass:: groco.layers.conv2d.GroupConv2D

.. autoclass:: groco.layers.conv3d.GroupConv3D

.. autoclass:: groco.layers.conv2d_transpose.GroupConv2DTranspose

.. autoclass:: groco.layers.conv3d_transpose.GroupConv3DTranspose

Pooling layers:

.. autoclass:: groco.layers.pooling.GroupMaxPooling1D

.. autoclass:: groco.layers.pooling.GroupMaxPooling2D

.. autoclass:: groco.layers.pooling.GroupMaxPooling3D

.. autoclass:: groco.layers.pooling.GroupAveragePooling1D

.. autoclass:: groco.layers.pooling.GroupAveragePooling2D

.. autoclass:: groco.layers.pooling.GroupAveragePooling3D

.. autoclass:: groco.layers.pooling.GlobalGroupMaxPooling1D

.. autoclass:: groco.layers.pooling.GlobalGroupMaxPooling2D

.. autoclass:: groco.layers.pooling.GlobalGroupMaxPooling3D

.. autoclass:: groco.layers.pooling.GlobalGroupAveragePooling1D

.. autoclass:: groco.layers.pooling.GlobalGroupAveragePooling2D

.. autoclass:: groco.layers.pooling.GlobalGroupAveragePooling3D

Internals
--------

.. autoclass:: groco.groups.group.Group
   :members:
   :show-inheritance:
