<h1> Central Code Repository for the Master Thesis of L.X. Driever </h1>

This repository provides the most important pieces of code developed as a part of my (L.X. Driever) masterthesis at <b>EPFL</b> and the <b>University of Tokyo</b>. For a copy of the thesis, please reach out to me at leonhard.driever@epfl.ch. Due to the confidentiality of the CFD software used as a part of the research, certain parts of the code have been ommitted from this public repository.

Here the different code groups provided in this repository are briefly discussed. Please note that the TTiME package is also available as a publically accessibly Python PyPI package, which can be installed using the command <i><b>pip install ttime</b></i>

## issho

This is the code related to time-based fluid-structure simulations. Due to the CFD solver confidentiality. The code for the FSSimulation class and the corresponding tests have been ommitted from this repository.

## Data collection

Here, the code is provided, which was used for data collection. Firstly, this includes the <b>mfoil</b> class, which is used to split airfoil profiles into sections. As shown in <i><b>mfoil_example.py</b></i>, the split airfoil sections can be rotated and saved as separate profiles.

<i><b>minimizers.py</b></i> provides the code used for root finding in control surface equilibrium prediction. This includes both a function used for root-finding for Chebyshev interpolations (such as those of the TTiME package), and for root-finding using CFD simulations. Due to confidentiality reasons, the code related to the running of each CFD simulation has been ommitted.

## Neural Networks

The wrapper classes and function, which build upon <i><b>Pytorch Lightning</b></i> to provide easy usage of neural networks, are provided in <i><b>NN_maker.py</b></i>. The provided example Jupyter notebook demonstrates how this code can be applied in practice.

## TTiME

Short for <i><b>Tensor Trains in Mathematics and Engineering</b></i>, the TTiME package provides code for low-rank Chebyshev interpolation. Here, a slightly extended version of that available on PyPI is provided.