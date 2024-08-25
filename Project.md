# PySeldonlib
![pyseldonlib](https://raw.githubusercontent.com/User-DK/pyseldon/main/res/logotext.png)
PySeldonlib is a Python Package for Opinion Dynamics Simulation, an extension of the [`Seldon Framework`](https://github.com/seldon-code/seldon). It provides:

- Tools for the simulation of various Opinion Dynamics Models like the classical DeGroot Model, Deffuant Model, Activity Driven Model, etc.
- Tools to create, manipulate, and study complex networks which are either randomly generated or provided by the user.
- A clean and robust interface for conducting the simulations.

## Opinion Dynamics

Opinion dynamics is a field of study within the realm of complex systems and sociophysics that explores how opinions, beliefs, and attitudes evolve and spread within social networks. It combines elements of physics, social science, and mathematics to understand the mechanisms driving opinion formation and change under various influences, such as personal convictions, peer pressure, media impact, and societal norms.

Our work contributes to this interdisciplinary field by providing robust tools for simulation and analysis, aiding in the understanding of complex opinion dynamics phenomena [`Seldon-Code`](https://github.com/seldon-code).

## DeGroot Model Example

The DeGroot model is a model of social influence. It describes how agents in a network can reach a consensus by updating their opinions based on the opinions of their neighbors. The DeGroot model is a simple model of social influence that has been studied extensively in the literature. It is used to model a wide range of social phenomena, such as the spread of information, the formation of opinions, and the emergence of social norms.

Below is an example of reaching consensus in a network using the DeGroot model. We will create a network of 20 agents with random opinions and random connections between them. We will then conduct the simulation.

### Initial Opinions

Initial opinions of the agents in the network in the range of [0,1] are shown below:

![Initial Opinions](https://github.com/User-DK/pyseldon/raw/main/visualisations/ouput_20_agents_10_connections_each/initial.png)

### Final Opinions

Final opinions of the agents in the network after the simulation are shown below:

![Final Opinions](https://github.com/User-DK/pyseldon/raw/main/visualisations/ouput_20_agents_10_connections_each/final.png)

We can conclude that the agents have reached a consensus after the simulation.

### Reference
- DeGroot, M. H. (1974). Reaching a Consensus. Journal of the American Statistical Association, 69(345), 118–121. https://doi.org/10.2307/2285509


### Usage

```python
import pyseldonlib

pyseldonlib.run_simulation_from_config_file(config_file_path = '/path/to/config/file')
```

```python
import pyseldonlib

model = pyseldonlib.DeGroot_Model(max_iterations=1000,
                               convergence_tol=1e-6,
                               rng_seed=120, 
                               other_settings=other_settings)

output_dir_path = str("./output")

model.run(output_dir_path)
```