import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from common.reward import CustomRewardAndDoneEnv
from common.preprocess import SkipFrame, ResizeEnv
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv


def get_env(stage_names, n_envs=16):
    def make_env(rank, stage_name):
        def _init():
            print(f"Initializing environment {rank} for stage: {stage_name}")

            seed = rank
            env = gym_super_mario_bros.make(stage_name)
            env.seed(seed)

            MOVEMENT = [["right"], ["right", "A"], ["left"], ["left", "A"]]
            env = JoypadSpace(env, MOVEMENT)
            env = CustomRewardAndDoneEnv(env)
            env = SkipFrame(env, skip=4)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeEnv(env, size=84)

            print(f"Environment {rank} for stage {stage_name} initialized successfully")
            return env

        return _init

    if n_envs != 1:
        print(f"Creating {n_envs} environments for each of {len(stage_names)} stages...")
        env = SubprocVecEnv([make_env(i, stage_name) for i in range(n_envs) for stage_name in stage_names], start_method="spawn")
    else:
        env = DummyVecEnv([make_env(0, stage_names[0])])

    print("Environments created, adding frame stack...")
    env = VecFrameStack(env, 4, channels_order="last")
    print("Environment setup complete!")

    return env
