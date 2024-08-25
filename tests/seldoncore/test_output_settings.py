import pyseldonlib
import pytest


def test_output_settings():
    settings = pyseldonlib.seldoncore.OutputSettings()
    assert settings.n_output_agents is None
    assert settings.n_output_network is None
    assert settings.print_progress is False
    assert settings.output_initial is True
    assert settings.start_output == 1
    assert settings.start_numbering_from == 0

    # set values
    settings.n_output_agents = 10
    settings.n_output_network = 5
    settings.print_progress = True
    settings.output_initial = False
    settings.start_output = 5
    settings.start_numbering_from = 1

    # check values
    assert settings.n_output_agents == 10
    assert settings.n_output_network == 5
    assert settings.print_progress == True
    assert settings.output_initial == False
    assert settings.start_output == 5
    assert settings.start_numbering_from == 1
