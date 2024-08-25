import pyseldonlib
import pytest


def test_deffuant_settings():
    settings = pyseldonlib.seldoncore.DeffuantSettings()
    assert settings.max_iterations == None
    assert settings.homophily_threshold == 0.2
    assert settings.mu == 0.5
    assert settings.use_network == False
    assert settings.use_binary_vector == False
    assert settings.dim == 1

    # set values
    settings.max_iterations = 100
    settings.homophily_threshold = 0.3
    settings.mu = 0.6
    settings.use_network = True
    settings.use_binary_vector = True
    settings.dim = 2

    # check values
    assert settings.max_iterations == 100
    assert settings.homophily_threshold == 0.3
    assert settings.mu == 0.6
    assert settings.use_network == True
    assert settings.use_binary_vector == True
    assert settings.dim == 2


if __name__ == "__main__":
    pytest.main([__file__])
