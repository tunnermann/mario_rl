import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from common.reward import CustomRewardAndDoneEnv
from common.preprocess import SkipFrame, ResizeEnv
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv


def get_env(stage_name, n_envs=16):
    def make_env(rank):
        def _init():
            print(f"Initializing environment {rank}")

            seed = rank
            env = gym_super_mario_bros.make(stage_name)
            env.seed(seed)

            MOVEMENT = [["right"], ["right", "A"]]
            env = JoypadSpace(env, MOVEMENT)
            env = CustomRewardAndDoneEnv(env)
            env = SkipFrame(env, skip=4)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeEnv(env, size=84)

            print(f"Environment {rank} initialized successfully")
            return env

        return _init

    print(f"Creating {n_envs} environments...")
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="spawn")
    print("Environments created, adding frame stack...")
    env = VecFrameStack(env, 4, channels_order="last")
    print("Environment setup complete!")

    return env
