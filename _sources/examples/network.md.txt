---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Example Usage of the Network Class from the pyseldonlib Package

```{code-cell} ipython3
# Import necessary modules
import pyseldonlib
```

```{code-cell} ipython3
# Initialize the Network object
network = pyseldonlib.Network(
    model_string="DeGroot",
    neighbour_list=[[1, 2], [0, 2], [0, 1], [4], [3]],
    weight_list=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1], [1]],
    direction="Incoming"
)
```

```{code-cell} ipython3
# Print the network details
print(f"Number of agents: {network.n_agents}")
print(f"Edges of 1st index agent: {network.n_edges(1)}")
print(f"Direction: {network.get_direction}")
print(f"Neighbour List: {network.get_neighbours(1)}")
print(f"Weight List: {network.get_weights(1)}")
```
