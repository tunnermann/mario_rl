import pandas as pd
import numpy as np

def consolidate_data():
    # Read the CSV file
    df = pd.read_csv('presentation/reward_log_a2c.csv')

    # Initialize lists to store consolidated data
    timesteps_consolidated = []
    rewards_consolidated = []

    # Process data in groups of 4
    for i in range(0, len(df), 4):
        group = df.iloc[i:i+4]
        avg_reward = group['reward'].mean()
        
        # Stop if average reward > 900
        # if avg_reward > 900:
        #     break
            
        timesteps_consolidated.append(group['timesteps'].iloc[-1]/4)
        rewards_consolidated.append(avg_reward)

    # Create consolidated dataframe    
    df_consolidated = pd.DataFrame({
        'timesteps': timesteps_consolidated,
        'reward': rewards_consolidated
    })

    # Save to new CSV
    df_consolidated.to_csv('presentation/reward_log_a2c_consolidated_full.csv', index=False)

def consolidate_dqn_data():
    # Read the CSV file
    df = pd.read_csv('presentation/updated_dqn_metrics.csv')

    # Initialize lists to store consolidated data
    timesteps_consolidated = []
    rewards_consolidated = []

    # Process data every 10000 timesteps
    step_size = 10000
    for timestep in range(0, df['timesteps'].max() + step_size, step_size):
        # Get data points within this timestep window
        mask = (df['timesteps'] >= timestep) & (df['timesteps'] < timestep + step_size)
        group = df[mask]
        
        if not group.empty:
            # Use the last timestep in the group and average reward divided by 4
            timesteps_consolidated.append(group['timesteps'].iloc[-1])
            rewards_consolidated.append(group['reward_mean'].mean() / 4)

    # Create consolidated dataframe    
    df_consolidated = pd.DataFrame({
        'timesteps': timesteps_consolidated,
        'reward_mean': rewards_consolidated
    })

    # Save to new CSV
    df_consolidated.to_csv('presentation/reward_dqn.csv', index=False)

def combine_graphs():
    # Read the CSV files
    df_a2c = pd.read_csv('presentation/reward_log_a2c_consolidated_full.csv')
    df_ppo = pd.read_csv('presentation/reward_log_ppo.csv')
    df_ppo_2 = pd.read_csv('presentation/reward_ppo_multi_level.csv')
    df_dqn = pd.read_csv('presentation/reward_dqn.csv')

    # Create figure and axis
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    # Plot both lines
    plt.plot(df_a2c['timesteps'], df_a2c['reward'], label='A2C')
    plt.plot(df_ppo['timesteps'], df_ppo['reward'], label='PPO')
    plt.plot(df_ppo_2['timesteps'], df_ppo_2['reward'], label='PPO Multi Level')
    plt.plot(df_dqn['timesteps'], df_dqn['reward_mean'], label='DQN')

    # Customize the plot
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title('A2C vs PPO vs PPO Multi Level vs DQN Training Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('presentation/combined_training.png')
    plt.close()

if __name__ == "__main__":
    combine_graphs()
    # consolidate_data()
    # consolidate_dqn_data()