import gym
import minihack
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directories exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define a Custom Agent
class CustomAgent:
    def __init__(self, 
                 learning_rate=0.1, 
                 gamma=0.99,
                 epsilon=1.0, 
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def choose_action(self, state, action_space):
        state_key = self._state_to_key(state)  # Convert state to a hashable key
        if np.random.rand() < self.epsilon:
            action = action_space.sample()  # Explore
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(action_space.n)
            action = np.argmax(self.q_table[state_key])  # Exploit
        print(f"Chosen action: {action}")  # Debug print
        return action

    def update(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Ensure action is an integer
        if isinstance(action, np.ndarray):
            action = action.item()  # Convert numpy array to scalar

        # Debug prints
        print(f"State key: {state_key}, Type: {type(state_key)}")
        print(f"Next State key: {next_state_key}, Type: {type(next_state_key)}")
        print(f"Q-table type: {type(self.q_table)}")

        # Initialize Q-table entries if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(env.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(env.action_space.n)

        # Perform Q-learning update
        q_predict = self.q_table[state_key][action]
        q_target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.learning_rate * (q_target - q_predict)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _state_to_key(self, state):
        if isinstance(state, dict):
            return tuple(sorted((k, tuple(v.flatten())) for k, v in state.items()))
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten())  # Convert ndarray to a flattened tuple
        else:
            return tuple(state)  # Fallback for other types

# Create Environment
env_name = 'MiniHack-Room-Random-15x15-v0'
env = gym.make(env_name)

# Set up Parameters for the Agent
learning_rate = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 100

# Create an Instance of the Agent
agent = CustomAgent(learning_rate=learning_rate, 
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min)

# Training Loop
total_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state, env.action_space)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

    # Save per-episode information
    with open(os.path.join(output_dir, 'episode_logs.txt'), 'a') as log_file:
        log_file.write(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}\n")

# Save the Performance Plot
try:
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'total_rewards.png'))
    plt.close()
    print(f"Plot saved as {os.path.join(output_dir, 'total_rewards.png')}")
except Exception as e:
    print(f"Failed to save plot: {e}")

# Save final Q-table to a file
try:
    with open(os.path.join(output_dir, 'q_table.txt'), 'w') as q_table_file:
        for state_key, q_values in agent.q_table.items():
            q_table_file.write(f"State: {state_key}, Q-values: {q_values.tolist()}\n")
    print(f"Q-table saved as {os.path.join(output_dir, 'q_table.txt')}")
except Exception as e:
    print(f"Failed to save Q-table: {e}")
    
print("Training complete. Results and logs saved.")
# Close the Environment
env.close()


