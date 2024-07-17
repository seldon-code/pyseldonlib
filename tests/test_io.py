import pyseldon.seldoncore as pd

network = pd.generate_fully_connected_degroot(n_agents=100, weight = None, seed = None)


pd.agents_to_file(network, "/home/parrot_user/Desktop/pyseldon/agents.txt")
print(type(network))