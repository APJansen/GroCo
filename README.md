# Group Convolutions in Keras with GroCo

See the standalone Colab notebook GroupConv_intro for an elaborate introduction to group convolutions:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/APJansen/GroupConv/blob/GroupConv_intro.ipynb)

This is a work in progress implementation of group convolutions in Keras.


Based on the book, lectures and notebooks on the geommetric deep learning by Micheal Bronstein, Joan Bruna, Taco Cohen and Petar Veličković, found [here](https://geometricdeeplearning.com), 
and of course also on [the original paper](http://proceedings.mlr.press/v48/cohenc16.html) on group convolutions by Taco Cohen and Max Welling.
All credit goes to them, and all mistakes are my own.

The colab notebook (click the button above) is meant to be pedagogical and I hope can be useful as an introduction to group convolutions. 
I recommend to go through it after or in parallel with the sources above. 
I go into more detail in some areas and less in other. 
In particular the implementation of a group convolution layer in Keras is explained in detail.

Implemented functionality:
- `GroupConv2D`: performs the group convolution, for any of the implemented groups. Includes automatic padding to maintain equivariance.
- `GroupMaxPooling2D` and `GroupAveragePooling2D`: does the same padding to maintain equivariance if necessary, then pools over the spatial part
- groups: p4m (discussed in the Colab notebook) and all its subgroups

The layers are intended to work as closely to Keras's as possible. Group convolutions with the same total kernel size take nearly 50% longer than regular ones. 
This can perhaps be improved, though it's not super obvious how.

This is enough functionality already to make fully group-equivariant/invariant networks, but it does still need more testing.

One thing I definitely want to add is pooling/subsampling not just on the grid but also on the full group.

Others that I might have a look at adding:
- transpose group convolutions
- adapt everything to 1D and 3D convolutions
- groups on a hexagonal (2D) lattice

Any feedback or questions are welcome.
