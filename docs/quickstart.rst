Quick Start Guide
=================

This guide will help you get started with the basics of using the `pyseldonlib` package.

Installation
------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    install

Usage
-----

.. code-block:: python

    import pyseldonlib

    pyseldonlib.run_simulation_from_config_file(
        config_file_path="path/to/config.toml",
        agent_file_path="path/to/agent.csv",
        network_file_path="path/to/network.csv",
        output_dir_path="path/to/output"
    )

This will run the simulation as per the configuration file and save the output files in the specified directory.

Configuration TOML file Format
------------------------------

.. code-block:: toml

    [simulation]
    model = "DeGroot" 
    rng_seed = 120

    [io]
    n_output_network = 20
    n_output_agents = 1
    print_progress = false 
    output_initial = true 
    start_output = 1 
    start_numbering_from = 0 

    [model]
    max_iterations = 20 

    [DeGroot]
    convergence = 1e-3

    [network]
    number_of_agents = 300
    connections_per_agent = 10

Specifications
--------------

- **[simulation]**
  - `model`: The model to run. Options are DeGroot, Deffuant, ActivityDriven, ActivityDrivenInertial.
  - `rng_seed`: Seed for random number generation. If left empty, a random seed is used.

- **[io]**
  - `n_output_network`: Number of iterations between writing network files.
  - `n_output_agents`: Number of iterations between writing agent opinion files.
  - `print_progress`: Whether to print the iteration time. Default is false.
  - `output_initial`: Whether to print the initial opinions and network file from step 0. Default is true.
  - `start_output`: Iteration number from which to start writing outputs.
  - `start_numbering_from`: Initial step number before the simulation starts. Default is 0.

- **[model]**
  - `max_iterations`: Maximum number of iterations. If not set, the maximum is infinite.

- **[DeGroot]**
  - `convergence`: Convergence threshold for the DeGroot model.

- **[network]**
  - `number_of_agents`: Number of agents in the network.
  - `connections_per_agent`: Number of connections each agent has.


