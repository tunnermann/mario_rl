import multiprocessing
from multi_level_ppo.env import get_env
from multi_level_ppo.callback import TrainAndLoggingCallback
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10

# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000
N_ENVS = 2  # Number of environments


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def main():
    print("Starting main function...")
    # Define multiple stages
    STAGE_NAMES = [
        "SuperMarioBros-1-1-v3",
        "SuperMarioBros-1-2-v3",
        "SuperMarioBros-1-3-v3",
        "SuperMarioBros-1-4-v3"
    ]

    TOTAL_TIMESTEP_NUMB = 6000000 * N_ENVS * len(STAGE_NAMES)
    CHECK_FREQ_NUMB = 20000 // (N_ENVS * len(STAGE_NAMES)) # Callback frequency

    env = get_env(STAGE_NAMES, N_ENVS)
    print("Environment created, testing reset...")

    # Test environment
    env.reset()
    print("Reset successful, creating model...")

    # policy_kwargs = dict(
    #     features_extractor_class=MarioNet,
    #     features_extractor_kwargs=dict(features_dim=512),
    #     optimizer_class=torch.optim.RMSprop,
    #     optimizer_kwargs=dict(eps=RMS_PROP_EPS, alpha=0.99, weight_decay=0.0),
    # )

    save_dir = Path("./model_ppo_multi_level")
    save_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = save_dir / "reward_log.csv"

    with open(reward_log_path, "a") as f:
        print("timesteps,reward,best_reward", file=f)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(
            optimizer_class=RMSpropTFLike,
            optimizer_kwargs=dict(eps=1e-5)
        ),
        tensorboard_log=save_dir,
        learning_rate=linear_schedule(LEARNING_RATE),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE,
        ent_coef=ENT_COEF,
        clip_range=0.2,
        clip_range_vf=0.2,
    )
    print("Model created, starting training...")
    # model = A2C.load(save_dir / 'best_model_2000000.zip', env=env)

    callback = TrainAndLoggingCallback(
        check_freq=CHECK_FREQ_NUMB,
        save_path=save_dir,
        episode_numbers=EPISODE_NUMBERS,
        env=env,
        max_timestep_test=MAX_TIMESTEP_TEST,
        model=model,
        total_timesteps=TOTAL_TIMESTEP_NUMB,
        reward_log_path=reward_log_path,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
