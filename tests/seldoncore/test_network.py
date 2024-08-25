import pytest
import pyseldonlib


def test_network_class_tests():
    # Generate some network
    n_agents = 20
    n_connections = 10
    gen_pseudorandom = 0

    network = pyseldonlib.seldoncore.generate_n_connections_(
        n_agents=n_agents,
        n_connections=n_connections,
        self_interaction = False,
    )

    assert network is not None
    assert network.n_agents() == n_agents
    assert network.n_edges() == n_agents * n_connections

    # Check that the function for setting neighbours and a single weight work
    # Agent 3
    agent_index = 3
    neighbour_list = [0, 10]
    weights = [0.5, 0.5]
    network.set_neighbours_and_weights(
        agent_idx=agent_index, buffer_neighbours=neighbour_list, buffer_weights=weights
    )

    retrieved_weights = network.get_weights(agent_index)
    assert retrieved_weights == weights

    # Checking that set_weight, get_neighbour work
    weights = [0.25, 0.55]
    network.set_weights(agent_idx=agent_index, weights=weights)
    buffer_w_get = network.get_weights(agent_index)

    assert buffer_w_get == weights
    assert network.n_edges(3) == 2
    assert neighbour_list[0] == network.get_neighbours(3)[0]
    assert network.get_weights(3)[1] == 0.55

    # Checking that set_neighbours_and_weights works with a vector of weights, push_back and transpose
    neighbour_list = [0, 10, 15]
    weights = [0.1, 0.2, 0.3]
    network.set_neighbours_and_weights(
        agent_idx=agent_index,
        buffer_neighbours=neighbour_list,
        buffer_weights=weights,
    )

    retrieved_weights = network.get_weights(agent_index)
    retrieved_neighbours = network.get_neighbours(agent_index)
    assert retrieved_weights == weights
    assert retrieved_neighbours == neighbour_list

    # Now we test the toggle_incoming_outgoing() function
    # First record all the old edges as tuples (i,j,w) where this edge goes from j -> i with weight w
    old_edges = set()

    for i in range(network.n_agents()):
        buffer_n = network.get_neighbours(i)
        buffer_w = network.get_weights(i)
        for j in range(len(buffer_n)):
            neigh = buffer_n[j]
            weight = buffer_w[j]
            edge = (i, neigh, weight)
            old_edges.add(edge)

    old_direction = network.direction()
    network.toggle_incoming_outgoing()
    new_direction = network.direction()
    # Check that the direction has changed
    assert old_direction != new_direction

    # Now we go over the toggled network and try to re-identify all edges
    for i in range(network.n_agents()):
        buffer_n = network.get_neighbours(i)
        buffer_w = network.get_weights(i)
        for j in range(len(buffer_n)):
            neigh = buffer_n[j]
            weight = buffer_w[j]
            edge = (neigh, i, weight)
            assert edge in old_edges
            old_edges.remove(edge)

    assert len(old_edges) == 0

    # Test remove double counting
    neighbour_list = [[2, 1, 1, 0], [2, 0, 1, 2], [1, 1, 0, 2, 1], [], [3, 1]]
    weights = [[-1, 1, 2, 0], [-1, 1, 2, -1], [-1, 1, 2, 3, 1], [], [1, 1]]

    neighbour_no_double_counting = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [], [1, 3]]
    weights_no_double_counting = [[0, 3, -1], [1, 2, -2], [2, 1, 3], [], [1, 1]]

    network = pyseldonlib.seldoncore.Network(neighbour_list, weights, "Incoming")
    network.remove_double_counting()

    for i_agent in range(network.n_agents()):
        assert (
            network.get_neighbours(i_agent) == neighbour_no_double_counting[i_agent]
        ), f"Neighbours mismatch for agent {i_agent}"
        assert (
            network.get_weights(i_agent) == weights_no_double_counting[i_agent]
        ), f"Weights mismatch for agent {i_agent}"

    # Test the generation of a square lattice neighbour list for three agents
    desired_neighbour_list = [
        [2, 1, 3, 6],
        [2, 0, 4, 7],
        [0, 1, 5, 8],
        [4, 5, 6, 0],
        [5, 3, 1, 7],
        [3, 4, 2, 8],
        [7, 8, 3, 0],
        [8, 6, 4, 1],
        [7, 6, 5, 2],
    ]
    network = pyseldonlib.seldoncore.generate_square_lattice_(n_edge=3)

    for i_agent in range(network.n_agents()):
        assert (
            network.get_neighbours(i_agent).sort()
            == desired_neighbour_list[i_agent].sort()
        ), f"Neighbours mismatch for agent {i_agent}"


# if __name__ == "__main__":
#     pytest.main([__file__])
