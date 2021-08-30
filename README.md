# GroupConv
Work in progress.

Exploring group convolutions, and implementing them as Keras layers.

Based on the book, lectures and notebooks on the geommetric deep learning by Micheal Bronstein, Joan Bruna, Taco Cohen and Petar Veličković, found [here](https://geometricdeeplearning.com), 
and of course also on [the original paper](http://proceedings.mlr.press/v48/cohenc16.html) on group convolutions by Taco Cohen and Max Welling.
All credit goes to them, and all mistakes are my own.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/APJansen/GroupConv/blob/GroupConv-Keras.ipynb)

The colab notebook (click the button above) is meant to be pedagogical and I hope can be useful as an introduction to group convolutions. 
I recommend to go through it after or in parallel with the sources above. 
I go into more detail in some areas and less in other. 
In particular the implementation of a group convolution layer in Keras is explained in detail.

For pedagogical purposes I think it is already useful in its current state. 
The implementation itself needs some work. 
It is already possible to create a fully equivariant/invariant model for the group we consider here, if one is careful with hyperparameters (see the strides section).
But it lacks other groups, subsampling to smaller groups, 
speed (it is 2-3 times slower than a similar sized network of regular convolutions), and more safeguards for incompatible hyperparameters.

Any feedback or questions are welcome.
