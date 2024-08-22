Quick Start Guide
=================

This guide will help you get started with the basics of using the `pyseldon` package.

Installation
------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    install

Usage
-----

.. code-block:: python

    import pyseldon

    pyseldon.run_simulation_from_config_file(config_file_path ="path/to/config.toml", agent_file_path="path/to/agent.csv", network_file_path="path/to/network.csv", output_dir_path="path/to/output")

TOML file Format
----------------

.. code-block:: toml

    [simulation]
    model = "" 
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


# The model to run. Options are DeGroot, Deffuant, DeffuantVector, ActivityDriven, Inertial
# The model specific parameters are given in the model section
# The network parameters are given in the network section
# The output parameters are given in the io section
# Leaving this empty will pick a random seed
# Write the network every 20 iterations
# Write the opinions of agents after every iteration
# Print the iteration time ; if not set, then does not prints
# Print the initial opinions and network file from step 0. If not set, this is true by default.
# Start writing out opinions and/or network files from this iteration. If not set, this is 1 + start_numbering_from.
# The initial step number, before the simulation runs, is this value. The first step would be (1+start_numbering_from). By default, 0
 # Model specific parameters
# If not set, max iterations is infinite
# If not set, the default 1e-6 is used