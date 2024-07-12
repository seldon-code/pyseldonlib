import pyseldon

initial_agents = []
final_agents = []

pyseldon.run_simulation_from_config_file(config_file_path='/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/DeGroot/conf.toml',initial_agents=initial_agents,final_agents=final_agents)

print(initial_agents)

print("------------------")

print(final_agents)