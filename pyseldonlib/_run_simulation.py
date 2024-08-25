from bindings import seldoncore
from typing import Optional


def run_simulation_from_config_file(
    config_file_path: str,
    agent_file_path: Optional[str] = None,
    network_file_path: Optional[str] = None,
    output_dir_path: Optional[str] = None,
):
    """Run the simulation using the configuration(toml) file.

    Parameters
    -----------
    config_file_path : str
        The path to the configuration(toml) file.

    agent_file_path : str, optional
        The path to the agent file.

    network_file_path : str, optional
        The path to the network file.

    output_dir_path : str, deafult="./output"
    """
    seldoncore.run_simulation(
        config_file_path=config_file_path,
        options=None,
        agent_file_path=agent_file_path,
        network_file_path=network_file_path,
        output_dir_path=output_dir_path,
    )


def run_simulation_from_options(
    options: object,
    agent_file_path: Optional[str] = None,
    network_file_path: Optional[str] = None,
    output_dir_path: Optional[str] = None,
):
    """
    Run the simulation using the simulation options object.

    Note
    ----
      The options object must be created using the SimulationOptions class.

    Parameters
    -----------
    options : object
        The simulation options object.
    agent_file_path : str, optional
        The path to the agent file.
    network_file_path : str, optional
        The path to the network file.
    output_dir_path : str, optional
        The path to the output directory.
    """
    seldoncore.run_simulation(
        options=options.options,
        config_file_path=None,
        agent_file_path=agent_file_path,
        network_file_path=network_file_path,
        output_dir_path=output_dir_path,
    )


def parse_config_file(file_path: str):
    """Parse the toml file and create the simulation options object.

    Parameters
    -----------
    file_path : str
        The path to the toml file.

    Returns
    -------
    Simulation: The simulation object.
    """
    return seldoncore.parse_config_file(file_path)
