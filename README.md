# Group Convolutions in Keras with GroCo

The aim of GroCo is to generalize the convolutional layers and all related functionality in Keras to group convolutions, following [the original paper on group convolutions by Taco Cohen and Max Welling](http://proceedings.mlr.press/v48/cohenc16.html), and to keep the interface as close as possible to the standard layers.

It was inspired by the book, lectures and notebooks on the geometric deep learning by Michael Bronstein, Joan Bruna, Taco Cohen and Petar Veličković, found [here](https://geometricdeeplearning.com), which I highly recommend.

I am not aware of any other Keras implementation. The implementation by the authors themselves is in Chainer/Tensorflow 1, [GrouPy](https://github.com/tscohen/GrouPy).
The intent is to not only translate this to Keras but also to expand on that functionalty.
I started this to learn, both about group convolutions and geometric deep learning more generally, but also about tensorflow and Keras, 
so any feedback is highly appreciated, you can reach me on twitter @aron_jansen.

# Introduction to group convolutions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/APJansen/GroupConv/blob/GroupConv_intro.ipynb)

This Colab notebook is a standalone introduction to group convolutions, meant to be read after/in parallel with the lectures, book or paper mentioned above.
It does not use the implementation in GroCo, but rather derives an early version of it from scratch, going into many of the nitty gritty aspects.
I hope it can be useful in parallel with the other sources.

# Example notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/APJansen/GroupConv/blob/example_MNIST.ipynb)

This Colab notebook illustrates how to use GroCo by constructing a group convolutional network, training it on MNIST and comparing to a regular network.
It also illustrates how to pool onto subgroups, which increases performance on MNIST. (Though not compared to the regular convolution, it is just used as a simple example but doesn't lend itself well to group convolutions as orientation matters in MNIST images.)


# Overview

Convolutions are _equivariant_ to translations, meaning if we have a convolutional layer `L`, an input `x` and a translation `T`, then if we translate the input and then apply the layer, we obtain the same result as if we apply the layer first, and then perform the translation: `L(T(x)) == T(L(x))`.
Group convolutions generalize this and are equivariant to a larger group.

All of the generalizations follow the same three step procedure:
1. Interpret the existing functionality in terms of a group. Most importantly, we interpret a grid (of pixels say) not as points but rather as translations. The mapping is trivial: the grid point with indices `(i, j)` is a translation that maps any point `(x, y)` to `(x + i, y + j)`.
2. Expand the group with what's called a point group, from just translations to translations combined with other symmetries such as rotations and mirroring along an axis. The original translations together with the new point group as a whole is called a wallpaper group.
3. Apply the standard functionality directly, in the same manner, to this larger group.

Here we give a quick overview of the implemented functionality. `G` refers to the point group, and `|G|` to its order, or the number of elements.
The largest wallpaper group on a square lattice, p4m, and all its subgroups have been implemented.

| -  | Convolution |
| ------------- | ------------- |
| Group Interpretation  | Apply group transformations to a kernel, multiply with signal and sum  |
| Generalization  | Before the regular convolution transform the kernel with the point group  |
| Implementation | `GroupConv2D(group='point_group_name', ...)` |
| Resulting differences | kernel gets copied `\|G\|` times, number of parameters stays the same but output channel grows by factor `\|G\|` |

| -  | stride |
| ------------- | ------------- |
| Group Interpretation  | subsample on a subgroup of the translations, i.e. the translations with even coordinates for stride 2 |
| Generalization  | subsampling onto any subgroup of the wallpaper group |
| Implementation | `GroupConv2D(..., stride=s, subgroup='point_subgroup_name', ...)` |
| Resulting differences | strides are done as usual*, and independently we can subsample on a subgroup of the point group |

(* strides are tricky in that they can cause the origin of the new, smaller grid to not coincide with the original origin and this breaks equivariance.
To prevent this the default `padding` option is `valid_equiv`, which pads a minimal extra amount to prevent this. Can be turned off by setting it back to `valid`, and `same_equiv` is also possible.)

| -  | pooling |
| ------------- | ------------- |
| Group Interpretation  | again subsample on subgroup of strides, but first aggregate on its cosets closest to the identity |
| Generalization  | subsampling onto any subgroup of the wallpaper group |
| Implementation | `GroupMaxPooling2D(group='group_name', subgroup='subgroup_name', ...)`, and the same with `GroupAveragePooling2D`|
| Resulting differences | in addition to pooling over the grid, potentially subsample on a subgroup of the point group, after aggregating on its cosets |


# Implemented groups
| name | symmetries |
| ------------- | ------------- |
| P4M | 90 degree rotations and mirroring on both axes |
| P4  | 90 degree rotations |
| P2MM | mirroring on both axes, 180 degree rotation |
| PMh | mirroring on horizontal axis |
| PMv | mirroring on vertical axis |
| P2 | 180 degree rotation |
| P1 | nothing |


Intended additions:
- The 1D and 3D versions of `GroupConv`, `GroupMaxPooling` and `GroupAveragePooling`
- `SeparableGroupConv`
- `GroupConv2DTranspose`
- `DepthwiseGroupConv2D`
- `GroupDense`, for when the group does not include translations
- something like `GroupReduce`, reducing equivariance to a subgroup, but rather than pooling keep all the features (merge with channel axis)
