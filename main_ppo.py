import torch
from pathlib import Path
import datetime

# Gym is an OpenAI toolkit for RL

# NES Emulator for OpenAI Gym

# Super Mario environment for OpenAI Gym


from ppo.mario_env import get_env
from ppo.mario_ppo import Mario_PPO
from ppo.ppo_logger import MetricLogger

env = get_env()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("ppo_checkpoints") / datetime.datetime.now().strftime(
    "%Y-%m-%dT%H-%M-%S"
)
save_dir.mkdir(parents=True)

mario = Mario_PPO(
    state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir
)
logger = MetricLogger(save_dir)
episode_policy_loss = 0
episode_value_loss = 0

# In the training loop:
episodes = 40000
for e in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    done = False

    # Reset episode metrics
    episode_reward = 0
    episode_length = 0  # This will count steps in the episode
    episode_policy_loss = 0
    episode_value_loss = 0
    n_updates = 0

    while not done:
        action = mario.act(state)
        next_state, reward, done, trunc, info = env.step(action)

        # Store transition
        mario.rewards.append(float(reward))
        mario.dones.append(done)

        # Update episode metrics
        episode_reward += reward
        episode_length += 1  # Increment step counter

        state = next_state

        if done or trunc or info["flag_get"]:
            break

    episode_policy_loss, episode_value_loss = mario.learn()
    mario.clear_memory()
    print(episode_reward, episode_policy_loss, episode_value_loss)

    # Calculate average losses for the episode
    if n_updates > 0:
        episode_policy_loss /= n_updates
        episode_value_loss /= n_updates

    # Log the episode with the correct length
    logger.log_step(
        reward=episode_reward,
        policy_loss=episode_policy_loss,
        value_loss=episode_value_loss,
        length=episode_length,  # Add this parameter
    )
    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, step=e)
    if e % 100 == 0:
        mario.save(e)
