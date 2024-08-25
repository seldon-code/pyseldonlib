"""This module provides the main interface to the pyseldonlib package.

It includes the following classes and functions:
- DeGrootModel
- Deffuant_Model
- Deffuant_Vector_Model
- Inertial_Model
- Activity_Driven_Model
- Other_Settings
- Network
- run_simulation_from_config_file
- run_simulation_from_options
- parse_config_file
"""

from typing import Optional, Union
from .DeGrootModel import DeGroot_Model
from .DeffuantModel import Deffuant_Model
from .DeffuantVectorModel import Deffuant_Vector_Model
from .InertialModel import Inertial_Model
from .ActivityDrivenModel import Activity_Driven_Model
from ._othersettings import Other_Settings
from .utils import *
from .network import Network
from ._run_simulation import (
    run_simulation_from_config_file,
    run_simulation_from_options,
    parse_config_file,
)
from bindings import seldoncore

__all__ = [
    "run_simulation_from_config_file",
    "run_simulation_from_options",
    "DeGroot_Model",
    "Deffuant_Model",
    "Deffuant_Vector_Model",
    "Inertial_Model",
    "Activity_Driven_Model",
    "Other_Settings",
    "Network",
    "parse_config_file",
]
