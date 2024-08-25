# pyseldonlib
![](https://raw.githubusercontent.com/User-DK/pyseldon/main/res/logotext.png)

## Python bindings for the Seldon framework

This work is a part of Google Summer of Code 2024 under the Python Software Organisation. The aim of this project is to create Python bindings for the Seldon framework. The Seldon framework is a C++ Engine for Opinion Dynamics. The bindings and the complete project as a python package will allow users to use the Seldon framework in Python and carry out Simulations.

### DeGroot Model Example
The DeGroot model is a model of social influence. It describes how agents in a network can reach a consensus by updating their opinions based on the opinions of their neighbors. The DeGroot model is a simple model of social influence that has been studied extensively in the literature. It is used to model a wide range of social phenomena, such as the spread of information, the formation of opinions, and the emergence of social norms.

And here is the example of reaching consensus in a network using the DeGroot model.
We will create a network of 20 agents with random opinions and random connections between them. We will then conduct the simulation. as can be seen in the notebook file [here](./examples/ouput_20_agents_10_connections_each/degrootmodel.ipynb).

Initial opinions of the agents in the network in the range of [0,1] are shown below:
![initial opinions](https://github.com/User-DK/pyseldon/raw/main/visualisations/ouput_20_agents_10_connections_each/initial.png)

Final opinions of the agents in the network after the simulation are shown below:
![final opinions](https://github.com/User-DK/pyseldon/raw/main/visualisations/ouput_20_agents_10_connections_each/final.png)

And we can conclude that the agents have reached a consensus after the simulation.

### Reference
- DeGroot, M. H. (1974). Reaching a Consensus. Journal of the American Statistical Association, 69(345), 118–121. https://doi.org/10.2307/2285509

visualisations generated with the help of `cytoscape` some of the visualisations normalisation code can be found [here](./visualisations/)

### Steps to Install and Compile the code

### Installation
Create a new `micromamba` environment and activate it:
```bash
micromamba create -f environment.yml
micromamba activate pyseldonenv
```
If you get problem with micromamba can look at the installation documentation [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

### Build
Set up the build directory with `meson`:
```bash
meson setup build
```

### Install Python package
Use `pip` to install the package:
```bash
pip install .
```

### Run the tests
Use `pytest` to run the tests:
```bash
pytest tests/
```
 or

 ```bash
pytest -s tests/
```
to see the output of the tests without capturing the std output

