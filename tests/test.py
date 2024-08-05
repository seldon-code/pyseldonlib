import pathlib as ptlb

# import os
import pytest
import pyseldon
import shutil


def test_run_simulation(capsys):
    # Set up the paths
    base_dir = ptlb.Path(__file__).parent.resolve()
    config_file = str(base_dir / "config/conf.toml")
    network_file = str(base_dir / "network/network.txt")
    invalid_config_file = str(base_dir / "config/inconf.txt")
    output_dir1 = str(base_dir / "outputs/outputfile")
    output_dir2 = str(base_dir / "outputs/opwithnetwork")
    output_dir = str(base_dir / "outputs/output")

    if ptlb.Path(output_dir1).exists():
        shutil.rmtree(output_dir1)
    if ptlb.Path(output_dir2).exists():
        shutil.rmtree(output_dir2)

    # Test with output directory and config
    with capsys.disabled():
        pyseldon.seldoncore.run_simulation(
            config_file_path=config_file, output_dir_path=output_dir1
        )
    assert ptlb.Path(output_dir1).exists()
    assert ptlb.Path(output_dir1).is_dir()

    # Test with network file
    with capsys.disabled():
        pyseldon.seldoncore.run_simulation(
            config_file_path=config_file,
            network_file_path=network_file,
            output_dir_path=output_dir2,
        )
    assert ptlb.Path(output_dir2).exists()
    assert ptlb.Path(output_dir2).is_dir()

    # Test with non-existent network file
    invalid_network_file = str(ptlb.Path(base_dir, "tests/network/net.txt"))
    with pytest.raises(RuntimeError):
        with capsys.disabled():
            pyseldon.seldoncore.run_simulation(
                config_file_path=config_file, network_file_path=invalid_network_file
            )

    # Test with invalid config file
    with pytest.raises(RuntimeError):
        with capsys.disabled():
            pyseldon.seldoncore.run_simulation(config_file_path=invalid_config_file)

    if ptlb.Path(output_dir).exists():
        shutil.rmtree(output_dir)


def test_settings():
    degroot_settings = pyseldon.seldoncore.DeGrootSettings()
    output_settings = pyseldon.seldoncore.OutputSettings()
    deffuant_settings = pyseldon.seldoncore.DeffuantSettings()
    activitydriven_settings = pyseldon.seldoncore.ActivityDrivenSettings()
    activitydriveninertial_settings = pyseldon.seldoncore.ActivityDrivenInertialSettings()
    initial_network_settings = pyseldon.seldoncore.InitialNetworkSettings()

    assert degroot_settings is not None
    assert output_settings is not None
    assert deffuant_settings is not None
    assert activitydriven_settings is not None
    assert activitydriveninertial_settings is not None
    assert initial_network_settings is not None
    assert activitydriveninertial_settings.covariance_factor == 0.0


# def test_network():
#     degroot_network = pyseldon.seldoncore.DeGrootNetwork()
#     deffuant_network = pyseldon.seldoncore.DeffuantNetwork()
#     activitydriven_network = pyseldon.seldoncore.ActivityDrivenNetwork()
#     activitydriveninertial_network = pyseldon.seldoncore.InertialNetwork()

#     assert degroot_network is not None
#     assert deffuant_network is not None
#     assert activitydriven_network is not None
#     assert activitydriveninertial_network is not None

# def test_simulation_with_simulationOptions():
#     degroot_settings = pyseldon.seldoncore.DeGrootSettings()
#     output_settings = pyseldon.seldoncore.OutputSettings()
#     initial_network_settings = pyseldon.seldoncore.InitialNetworkSettings()
#     simulation_options = pyseldon.seldoncore.SimulationOptions()
#     simulation_options.output_settings = output_settings
#     simulation_options.model_settings = degroot_settings
#     simulation_options.network_settings = initial_network_settings
#     simulation_options.model_string = "DeGroot"

#     base_dir = ptlb.Path(__file__).parent.resolve()
#     output_dir = str(base_dir / "outputs/output")

#     pyseldon.seldoncore.run_simulation(options = simulation_options,output_dir_path = output_dir)
#     assert ptlb.Path(output_dir).exists()
#     shutil.rmtree(output_dir)

if __name__ == "__main__":
    pytest.main([__file__])
