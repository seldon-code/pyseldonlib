import pyseldon
import pathlib

# pyseldon.seldoncore.run_simulation(config_file_path='/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDriven/conf.toml')

# sim_options = pyseldon.seldoncore.SimulationOptions(model_string = "Deffuant", model_settings = pyseldon.seldoncore.DeffuantSettings(max_iterations = 20), output_settings = pyseldon.seldoncore.OutputSettings(n_output_agents= 200,n_output_network = 5))
# simulation = pyseldon.seldoncore.SimulationSimpleAgent(sim_options)
# agent = simulation.network.agent[1].data.opinion

# print(agent)
# print(type(agent))

# simulation = pyseldon.seldoncore.SimulationSimpleAgent(pyseldon.seldoncore.parse_config_file("/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/DeGroot/conf.toml"))
# agent = simulation.network.agent

# simulation.run()
# for i in agent: 
#   print(i.data.opinion)
# print(dir(simulation))

# pyseldon.seldoncore.run_simulation(config_file_path = "/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDrivenBot/conf.toml")

# agent = simulation.network.agent[5].data.opinion
# print(agent)


# print(type(DeGrootSettings()))
# print(type(Network()))

# network = Network(model_string="DeGroot", n_agents=5)
# print(network.n_agents)    
# op = OutputSettings(n_output_agents = 1)
# op.n_output_agents = 2
# # print(op.print_settings)

# deg = DeGrootSettings(max_iterations =1)
# deff = DeffuantSettings(max_iterations = 2, homophily_threshold=0.5)
# deffv = DeffuantSettings(max_iterations = 2, homophily_threshold=0.5, use_binary_vector=True, dim=2)
# # print(deg)
# # print(deg.settings.max_iterations)
# # print(deg.max_iterations)

# sett = SimulationOptions(model_string = "DeGroot", output_settings=op, model_settings=deg)
# print(sett.model_string)
# print(sett.model)
# print(sett.output_settings)
# print(sett.options)

# x = Simulation(options = SimulationOptions(model_string = "DeffuantVector", output_settings=op, model_settings= deffv), agent_file_path=None, network_file_path=None)
# # x.run()
# network = x.network
# print(network.n_agents())
# print(network.n_agents)

model = pyseldon.DeGrootModel(max_iterations=100)
model.run()