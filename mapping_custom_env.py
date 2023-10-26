import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from gymnasium.spaces import Discrete, MultiDiscrete

DEBUG = False

class CustomRLEnvironment(gym.Env):
    
    def __init__(self, P, M, node_capacity, adj_matrix, n_msgs):
        super(CustomRLEnvironment, self).__init__()
        self.seed = 0
        self.P = P
        self.M = M
        # Save init array for environment reset
        self.node_capacity = self.node_capacity_init = node_capacity
        self.adj_matrix = adj_matrix
        self.n_msgs = n_msgs

        # Initialize all processes unassigned
        self.current_assignment = np.full(self.P, (self.M + 1))
        self.action_space = Discrete(self.P * self.M)
        self.observation_space = MultiDiscrete([self.M + 2] * self.P)

        #FIXME observation space es correcto?
        # self.observation_space = gym.spaces.MultiDiscrete(
        #     [self.M + 2 for _ in range(self.P)]
        # )

    def reset(self, seed=None, options=None):
        # FIXME new episode. Should start from last done state or re-init values? 
        super().reset(seed=seed, options=options)
        self.node_capacity = self.node_capacity_init.copy()  # Reset node capacities
        self.current_assignment = np.full(self.P, (self.M + 1))
        return self.current_assignment, {}

    def step(self, action):

        process_id, node_id = action // self.M, action % self.M
        
        # TODO Truncate when cannot place
        cannot_place = (self.node_capacity[node_id] == 0) or (self.current_assignment[process_id] != (self.M+1))

        # If a node cant place a process, reward is 0
        if cannot_place:
            reward = 0
        else:
            reward = count_communications(self.current_assignment, self.adj_matrix, (self.M+1))
            self.node_capacity[node_id] -= 1
            self.current_assignment[process_id] = node_id


        # Check if all processes have been assigned and done with an episode
        full_capacity = np.all(self.node_capacity == 0)
        all_assigned = np.all(self.current_assignment != (self.M + 1))
        done = full_capacity or all_assigned

        info = get_info(action, self.current_assignment, self.node_capacity, reward, done, full_capacity, all_assigned)
        
        return self.current_assignment, reward, done, False, info

    def render(self, mode="human"):
        # fig, ax = plt.subplots()
        # nx.draw(self.graph, pos=self.node_positions, with_labels=True, ax=ax)
        # plt.show()
        pass
        
    def close(self):
        # Clean up resources if needed
        pass


def count_communications(positions, adjacency_matrix, not_assigned):

    positions_nulled = positions.copy()
    positions_nulled[positions_nulled==not_assigned] = -1
    positions_nulled += 1

    # Create a mask for positions where processes are different
    mask_0 = np.logical_and(positions_nulled[:, None], positions_nulled)
    mask_1 = positions_nulled[:, None] != positions_nulled
    
    mask = mask_0 & mask_1

    # Use the mask to filter the adjacency_matrix
    communications_matrix = adjacency_matrix * mask

    # Count the number of non-zero elements (corresponding to communications) in the communications_matrix
    communications_count = np.count_nonzero(communications_matrix)

    return 1/(communications_count+1)

def get_info(action, current_assignment, node_capacity, reward, done, full_capacity, all_assigned):
    return {"Action": action,
    "Current Assignment": current_assignment,
    "Node Capacities": node_capacity,
    "Reward": reward,
    "Done": done,
    "Full nodes": full_capacity,
    "All procs assigned": all_assigned}


# G = nx.Graph()
# G.add_nodes_from