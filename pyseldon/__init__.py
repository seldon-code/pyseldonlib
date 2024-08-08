"""This module provides the main interface to the pyseldon package."""

from typing import Optional, Union
from .DeGrootModel import DeGrootModel
from .DeffuantModel import Deffuant_Model
from .DeffuantVectorModel import Deffuant_Vector_Model
from .InertialModel import Inertial_Model
from .ActivityDrivenModel import Activity_Driven_Model
from ._othersettings import Other_Settings
from ._network import Network
from ._run_simulation import (
    run_simulation_from_config_file,
    run_simulation_from_options,
    parse_config_file,
)


__all__ = ["run_simulation_from_config_file", "run_simulation_from_options"]
