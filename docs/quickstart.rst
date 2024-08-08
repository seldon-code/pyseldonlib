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

    # run the simulation from a configuration file

    pyseldon.run_simulation_from_config_file(config_file_path ="path/to/config.toml", agent_file_path="path/to/agent.csv", network_file_path="path/to/network.csv", output_dir_path="path/to/output")

TOML file Format
----------------

.. code-block:: toml

    
