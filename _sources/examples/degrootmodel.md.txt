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

# DeGroot Model (Reaching Consensus)

+++

Import the packages

```{code-cell} ipython3
import pyseldonlib
import pathlib
import shutil
```

Initialize some other settings for the simulation

```{code-cell} ipython3
other_settings = pyseldonlib.Other_Settings(n_output_agents=10,
                                         n_output_network= None, 
                                         print_progress= True, 
                                         output_initial=True, 
                                         start_output=1, 
                                         number_of_agents = 200, 
                                         connections_per_agent = 10)
```

Initialize the model with parameters.

```{code-cell} ipython3
model = pyseldonlib.DeGroot_Model(max_iterations=1000,
                               convergence_tol=1e-6,
                               rng_seed=120, 
                               other_settings=other_settings)
output_dir_path = str("./output")
if pathlib.Path(output_dir_path).exists():
  shutil.rmtree(output_dir_path)
model.run(output_dir_path)
if pathlib.Path(output_dir_path).exists():
  shutil.rmtree(output_dir_path)
```
### On your local machine you can see a C++ std::out