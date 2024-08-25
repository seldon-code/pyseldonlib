import pyseldonlib
import pytest


def test_activity_driven_settings():
    settings = pyseldonlib.seldoncore.ActivityDrivenSettings()
    assert settings.max_iterations == None
    assert settings.dt == 0.01
    assert settings.m == 10
    assert settings.eps == 0.01
    assert settings.gamma == 2.1
    assert settings.alpha == 3.0
    assert settings.homophily == 0.5
    assert settings.reciprocity == 0.5
    assert settings.K == 3.0
    assert settings.mean_activities is False
    assert settings.mean_weights is False
    assert settings.n_bots == 0
    assert settings.bot_m == []
    assert settings.bot_activity == []
    assert settings.bot_opinion == []
    assert settings.bot_homophily == []
    assert settings.use_reluctances is False
    assert settings.reluctance_mean == 1.0
    assert settings.reluctance_sigma == 0.25
    assert settings.reluctance_eps == 0.01
    assert settings.covariance_factor == 0.0

    # set values
    settings.max_iterations = 100
    settings.dt = 0.02
    settings.m = 20
    settings.eps = 0.02
    settings.gamma = 2.2
    settings.alpha = 3.1
    settings.homophily = 0.6
    settings.reciprocity = 0.6
    settings.K = 4.0
    settings.mean_activities = True
    settings.mean_weights = True
    settings.n_bots = 1
    settings.bot_m = [10]
    settings.bot_activity = [0.5]
    settings.bot_opinion = [0.5]
    settings.bot_homophily = [0.5]
    settings.use_reluctances = True
    settings.reluctance_mean = 1.1
    settings.reluctance_sigma = 0.26
    settings.reluctance_eps = 0.02
    settings.covariance_factor = 0.1

    # check values
    assert settings.max_iterations == 100
    assert settings.dt == 0.02
    assert settings.m == 20
    assert settings.eps == 0.02
    assert settings.gamma == 2.2
    assert settings.alpha == 3.1
    assert settings.homophily == 0.6
    assert settings.reciprocity == 0.6
    assert settings.K == 4.0
    assert settings.mean_activities is True
    assert settings.mean_weights is True
    assert settings.n_bots == 1
    assert settings.bot_m == [10]
    assert settings.bot_activity == [0.5]
    assert settings.bot_opinion == [0.5]
    assert settings.bot_homophily == [0.5]
    assert settings.use_reluctances is True
    assert settings.reluctance_mean == 1.1
    assert settings.reluctance_sigma == 0.26
    assert settings.reluctance_eps == 0.02
    assert settings.covariance_factor == 0.1


if __name__ == "__main__":
    pytest.main([__file__])
