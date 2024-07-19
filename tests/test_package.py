import pyseldon
import pathlib

# pyseldon.seldoncore.run_simulation(config_file_path='/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDriven/conf.toml')

# sim_options = pyseldon.seldoncore.SimulationOptions(model_string = "Deffuant", model_settings = pyseldon.seldoncore.DeffuantSettings(max_iterations = 20), output_settings = pyseldon.seldoncore.OutputSettings(n_output_agents= 200,n_output_network = 5))
# simulation = pyseldon.seldoncore.SimulationSimpleAgent(sim_options)
# agent = simulation.network.agent[1].data.opinion

# print(agent)
# print(type(agent))

simulation = pyseldon.seldoncore.SimulationSimpleAgent(pyseldon.seldoncore.parse_config_file("/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/DeGroot/conf.toml"))
agent = simulation.network.agent

simulation.run()
for i in agent: 
  print(i.data.opinion)
print(dir(simulation))

# pyseldon.seldoncore.run_simulation(config_file_path = "/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDrivenBot/conf.toml")

# agent = simulation.network.agent[5].data.opinion
# print(agent)