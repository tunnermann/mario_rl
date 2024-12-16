from pathlib import Path
import time
import gym
from stable_baselines3 import PPO

from multi_level_ppo.env import get_env
from common.model import MarioNet


def play(episodes=1):
    """
    When you want to watch Mario play
    """
    STAGE_NAME = ['SuperMarioBros-3-4-v3']
    env = get_env(STAGE_NAME, n_envs=1)

    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: True)

    # policy_kwargs = dict(
    #     features_extractor_class=MarioNet,
    #     features_extractor_kwargs=dict(features_dim=512),
    # )

    save_dir = Path('./model_ppo_multi_level')
    model_path = save_dir / 'best_model_750000'

    model = PPO.load(model_path, env=env)

    for _ in range(episodes):
        current_state = env.reset()

        done = False

        while not done:    
            time.sleep(0.02)
            env.render()
            action, _ = model.predict(current_state)
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
        
        # Close video recording
        env.close()

        break


if __name__ == "__main__":
    play()