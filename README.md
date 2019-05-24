# NN-Topology

The maximum width of a neural network imposes a fundamental limit on the topological complexity of the data that it can successfully classify.

The minimum required width for successful classification is equal to the dimension of the smallest dimensional space in which there exists a homeomorphism which renders an embedding of the input data linearly separable between classes.

Two two dimensional bows can be separated in two dimensions...

<img src="https://github.com/jcvdwlt/NN-Topology/blob/master/figs/2d_lines.gif">

but 2-D doughnut with a ball inside cannot...

<img src="https://github.com/jcvdwlt/NN-Topology/blob/master/figs/2d_doughnut.gif">

adding a dimension to one layer makes it trivial.

<img src="https://github.com/jcvdwlt/NN-Topology/blob/master/figs/3d_doughnut.gif">
