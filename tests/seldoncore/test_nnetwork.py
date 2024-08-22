from pyseldon import seldoncore

network = seldoncore.generate_square_lattice_activity_agent(10, 3.0)
seldoncore.network_to_dot_file_activity_agent(network, "network.txt")
seldoncore.agents_to_file_activity_agent(network, "agents.txt")