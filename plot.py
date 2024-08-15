import os
import re
import matplotlib.pyplot as plt

# Directory containing the text files
folder_path = 'output'  # Change this to your folder path

# Initialize lists to store data
all_episodes = []
all_rewards = []
all_epsilon = []

# Regular expression patterns for extracting data
reward_pattern = re.compile(r'Total Reward = (-?\d+\.\d+)')
epsilon_pattern = re.compile(r'Epsilon = (\d+\.\d+)')

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            episodes = []
            rewards = []
            epsilon = []
            
            # Process each line
            for line in lines:
                match_episode = re.match(r'Episode (\d+):', line)
                if match_episode:
                    episode = int(match_episode.group(1))
                    episodes.append(episode)
                
                reward_match = reward_pattern.search(line)
                if reward_match:
                    rewards.append(float(reward_match.group(1)))
                
                epsilon_match = epsilon_pattern.search(line)
                if epsilon_match:
                    epsilon.append(float(epsilon_match.group(1)))
                    
            all_episodes.append(episodes)
            all_rewards.append(rewards)
            all_epsilon.append(epsilon)

# Plot Total Rewards
plt.figure(figsize=(12, 6))
for i, rewards in enumerate(all_rewards):
    plt.plot(all_episodes[i], rewards, label=f'Agente {i + 1}')
plt.xlabel('Episódios')
plt.ylabel('Recompensas totais')
plt.title('Recompensas totais por Episódio (Parâmetros 3)')
plt.legend()
plt.grid(True)
plt.savefig('total_rewards.png')
plt.show()

# Plot Epsilon Values
plt.figure(figsize=(12, 6))
for i, epsilons in enumerate(all_epsilon):
    plt.plot(all_episodes[i], epsilons, label=f'Agente {i + 1}')
plt.xlabel('Episódios')
plt.ylabel('Epsilon')
plt.title('Epsilon Values per Episode from Different Files')
plt.legend()
plt.grid(True)
plt.savefig('epsilon_values.png')
plt.show()
