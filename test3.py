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
                 action_space,
                 learning_rate=0.1, 
                 gamma=0.95,
                 epsilon=1.0, 
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def choose_action(self, state):
        state_key = self._state_to_key(state)  # Convert state to a hashable key
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()  # Explore
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_space.n)
            action = np.argmax(self.q_table[state_key])  # Exploit
        return action

    def update(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Ensure action is an integer
        if isinstance(action, np.ndarray):
            action = action.item()  # Convert numpy array to scalar

        # Initialize Q-table entries if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)

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

def run_experiment(run_id, num_episodes):
    # Create Environment
    env_name = 'MiniHack-Room-Random-15x15-v0'
    env = gym.make(env_name)

    # Set up Parameters for the Agent
    learning_rate = 0.3
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # Create an Instance of the Agent with action_space
    agent = CustomAgent(action_space=env.action_space,
                        learning_rate=learning_rate, 
                        gamma=gamma,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min)

    # Training Loop
    total_rewards = []

    for episode in range(num_episodes):
        try:
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            total_rewards.append(total_reward)
            print(f"Run {run_id}, Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

            # Save per-episode information
            with open(os.path.join(output_dir, f'episode_logs_run_{run_id}.txt'), 'a') as log_file:
                log_file.write(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}\n")

        except Exception as e:
            print(f"Episode {episode + 1} failed due to: {e}")
            continue  # Continue to the next episode

    # Save the Performance Plot
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(total_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Total Rewards per Episode - Run {run_id}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'total_rewards_run_{run_id}.png'))
        plt.close()
        print(f"Plot for run {run_id} saved as {os.path.join(output_dir, f'total_rewards_run_{run_id}.png')}")
    except Exception as e:
        print(f"Failed to save plot for run {run_id}: {e}")

    # Save final Q-table to a file
    try:
        with open(os.path.join(output_dir, f'q_table_run_{run_id}.txt'), 'w') as q_table_file:
            for state_key, q_values in agent.q_table.items():
                q_table_file.write(f"State: {state_key}, Q-values: {q_values.tolist()}\n")
        print(f"Q-table for run {run_id} saved as {os.path.join(output_dir, f'q_table_run_{run_id}.txt')}")
    except Exception as e:
        print(f"Failed to save Q-table for run {run_id}: {e}")

    # Close the Environment
    env.close()

def main():
    num_runs = 5  # Number of times to run the experiment
    num_episodes = 100  # Number of episodes per run

    for run_id in range(1, num_runs + 1):
        try:
            print(f"Starting run {run_id}...")
            run_experiment(run_id, num_episodes)
            print(f"Run {run_id} complete.")
        except Exception as e:
            print(f"Run {run_id} failed due to: {e}")
            # Optionally handle the failure (e.g., log it, restart with different parameters, etc.)
            continue  # Ensure the script continues to the next run

if __name__ == "__main__":
    main()
