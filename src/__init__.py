"""This module provides the main interface to the pyseldon package."""

from .simulation import run_simulation_from_config_file, run_simulation_from_options

__all__ = ["run_simulation_from_config_file", "run_simulation_from_options"]
