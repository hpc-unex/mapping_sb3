import json
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN

from mapping_custom_env import CustomRLEnvironment



# Load the JSON data from the file
with open('./binomial_4_2.json', 'r') as file:
    data = json.load(file)

# Extract the relevant information
P = data["Graph"]["P"]
M = data["Graph"]["M"]
node_names = data["Graph"]["node_names"]
edges = data["Graph"]["comms"]["edges"]
volume = data["Graph"]["comms"]["volume"]
n_msgs = data["Graph"]["comms"]["n_msgs"]

# Extract node capacities from the JSON data
node_capacity = data["Graph"]["capacity"]

np_node_capacity = np.array(node_capacity)

# Extract the number of nodes (M)
M = data["Graph"]["M"]

# Initialize the adjacency matrix with zeros
adj_matrix = np.zeros((P, P))

# Populate the adjacency matrix with edge weights (volume)
edges = data["Graph"]["comms"]["edges"]
volume = data["Graph"]["comms"]["volume"]


for edge, msg_volume in zip(edges, volume):
    node1, node2 = edge
    adj_matrix[node1][node2] = msg_volume
    adj_matrix[node2][node1] = msg_volume  # Since it's an undirected graph


# Create the Gym environment with the adjacency matrix and node capacities
env = CustomRLEnvironment(P, M, np_node_capacity, adj_matrix, n_msgs)
model = DQN("MlpPolicy", env, verbose=1, learning_starts=5000, device='cpu')
model.learn(total_timesteps=10000, log_interval=4)

# Example usage:
obs = env.reset(4)
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Replace this with your RL agent's action selection logic
    next_obs, reward, done, info, _ = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)





