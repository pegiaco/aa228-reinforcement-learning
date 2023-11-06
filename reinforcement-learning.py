import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Create a Q-learning agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)


def run_q_learning(dataset_file, num_states, num_actions, learning_rate, discount_factor, epsilon, num_episodes):
    start_time = time.time()  # Record the start time

    # Read transition data from the provided dataset file
    data = pd.read_csv(f"./data/{dataset_file}")

    # Initialize the Q-learning agent
    agent = QLearningAgent(num_states, num_actions, learning_rate, discount_factor, epsilon)

    # Q-learning algorithm with a progress bar for the specified number of episodes
    for episode in tqdm(range(num_episodes), desc=f"Q-Learning Progress - {dataset_file.split('.')[0]} dataset"):
        for _, row in data.iterrows():
            current_state = int(row['s']) - 1  # Adjust to 0-based indexing
            action = int(row['a']) - 1  # Adjust to 0-based indexing
            # Ensure that the action is within the valid range (0 to num_actions-1)
            action = max(0, min(action, num_actions - 1))
            reward = float(row['r'])
            next_state = int(row['sp']) - 1  # Adjust to 0-based indexing

            agent.update_q_table(current_state, action, reward, next_state)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time

    # Save the optimal policy as a .policy file without column names with 1-based indexing
    optimal_policy = [agent.select_action(state) + 1 for state in range(num_states)]
    output_filename = f"{dataset_file.split('.')[0]}.policy"
    pd.DataFrame({'Action': optimal_policy}).to_csv(f"./output/{output_filename}", index=False, header=False)

    print(f"Optimal policy saved to ./output/{output_filename}")
    
    # Write dataset name and training time to a file
    with open("./output/training_times.txt", "a") as time_file:
        time_file.write(f"{dataset_file}: {training_time} seconds\n")



if __name__ == "__main__":
    num_episodes = 1000

    # Clear the training_times.txt file
    with open("./output/training_times.txt", "w") as time_file:
        time_file.write(f"num_episodes: {num_episodes}\n")

    run_q_learning("small.csv", num_states=100, num_actions=4, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, num_episodes=num_episodes)

    run_q_learning("medium.csv", num_states=50000, num_actions=7, learning_rate=0.1, discount_factor=1.0, epsilon=0.1, num_episodes=num_episodes)

    run_q_learning("large.csv", num_states=312020, num_actions=9, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, num_episodes=num_episodes)

