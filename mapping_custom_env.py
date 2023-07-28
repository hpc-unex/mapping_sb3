import numpy as np
import gymnasium as gym

class CustomRLEnvironment(gym.Env):
    def __init__(self, P, M, node_capacity, adj_matrix, n_msgs):
        super(CustomRLEnvironment, self).__init__()
        self.seed =0
        self.P = P
        self.M = M
        self.node_capacity = node_capacity
        self.adj_matrix = adj_matrix
        self.n_msgs = n_msgs
        # self.current_assignment = np.zeros(P, dtype=int)  # Initialize all processes unassigned
        self.current_assignment = np.full(self.P,self.M+1)  # Initialize all processes unassigned
        self.action_space = gym.spaces.Discrete(self.P * self.M)

        self.observation_space = gym.spaces.MultiDiscrete([self.M+2 for _ in range(self.P)])
    
        # Convert multi-dimensional actions to a single integer

    

    def reset(self, seed=None):
        self.current_assignment = np.full(self.P,self.M+1)
        self.node_capacity = self.node_capacity.copy()  # Reset node capacities
        return (self.current_assignment.copy()), None

    def step(self, action):
        process_id, node_id = action // self.M, action % self.M
        node_id = node_id
        # print("Proceso: ", process_id, " -> nodo: ", node_id)

        # Check if the node has enough capacity to accommodate the process
        if self.node_capacity[node_id] > 0:
            # Check if the process is already assigned to a node and release that capacity
            if self.current_assignment[process_id] != self.M+1:
                old_node_id = self.current_assignment[process_id]
                self.node_capacity[old_node_id] += 1  # Release the capacity of the previously assigned node

            self.current_assignment[process_id] = node_id
            self.node_capacity[node_id] -= 1  # Reduce the capacity of the assigned node

        # Calculate the total communication volume after this assignment
        total_volume = 0

        # Calculate the reward (negative total communication volume to minimize it)
        reward = count_communications(self.current_assignment, self.adj_matrix)
        
        # Check if all processes have been assigned and done with an episode
        done = np.all(self.node_capacity == 0) or np.all(self.current_assignment != 9)

        # Additional information for debugging or analysis
        info = {"total_volume": total_volume}
        # print("Capacidad de los nodos: ", self.node_capacity)
        # print("Current: ",self.current_assignment)
        # print("Recompensa: ", reward)
        # print("------------------------------------------------")
        return (self.current_assignment.copy()), reward, done, False, info

    def render(self, mode='human'):
        # You can implement rendering the environment if needed
        pass

    def close(self):
        # Clean up resources if needed
        pass
    

def count_communications(positions, adjacency_matrix):
        # Get the number of processes based on the length of the positions array
        num_processes = len(positions)

        # Create a mask for positions where processes are different
        mask = positions[:, None] != positions

        # Use the mask to filter the adjacency_matrix
        communications_matrix = adjacency_matrix * mask

        # Count the number of non-zero elements (corresponding to communications) in the communications_matrix
        communications_count = np.count_nonzero(communications_matrix)

        return communications_count