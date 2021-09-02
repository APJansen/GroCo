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


The implementation itself needs some work, but the group convolution layer itself is already functional for the most common groups.
It still needs pooling layers, more testing, and can probably be sped up more (it is currently 2-3 times slower than a similar sized network of regular convolutions).
It is already possible though to create a fully equivariant/invariant (on the last layer) model using strided convolutions rather than pooling to subsample.

Any feedback or questions are welcome.
