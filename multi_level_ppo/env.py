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

    # Create n_envs environments for each stage
    total_envs = n_envs * len(stage_names)
    env_creators = []
    
    for stage_name in stage_names:
        for i in range(n_envs):
            rank = len(env_creators)
            env_creators.append(make_env(rank, stage_name))
    
    print(f"Creating {n_envs} environments for each of {len(stage_names)} stages...")
    if total_envs != 1:
        env = SubprocVecEnv(env_creators, start_method="spawn")
    else:
        env = DummyVecEnv(env_creators)

    print("Environments created, adding frame stack...")
    env = VecFrameStack(env, 4, channels_order="last")
    print("Environment setup complete!")

    return env
