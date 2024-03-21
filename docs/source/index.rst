.. Groco documentation master file, created by
   sphinx-quickstart on Wed Feb 16 14:21:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======
GroCo
=====
Group equivariant Convolutions
------------------------------

GroCo is a package for group equivariant convolutions.
These are a special type of convolutional layers that respect a given set of symmetries.

As a simple example, consider a possible symmetry of reflections along the y-axis.
A layer that is equivariant with respect to this symmetry has the property that if you feed it an 
image and then flip the layer's output along the y-axis,
the result is the same as flipping the input image along the y-axis and then feeding it to the layer.

Another example is the symmetry of rotations by 90 degrees. These two symmetries can be combined
to produce more symmetries, forming the mathematical structure of a group.
All these groups that are compatible with a regular grid are implemented in GroCo.
An overview of these groups groups can be found in the :ref:`symmetries` section.

.. toctree::
   :maxdepth: 2
   :hidden:

   symmetries

   theory

   api

