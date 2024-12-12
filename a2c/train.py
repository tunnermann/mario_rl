import multiprocessing
from a2c.env import get_env
from a2c.callback import TrainAndLoggingCallback
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# Model Param
CHECK_FREQ_NUMB = 10000
LEARNING_RATE = 0.0007
ENT_COEF = 0.05
N_STEPS = 128  # Shorter rollouts for A2C
GAMMA = 0.99
N_ENVS = 4  # Number of environments
TOTAL_TIMESTEP_NUMB = 6000000 * N_ENVS

VF_COEF = 0.5  # Value Function coefficient
RMS_PROP_EPS = 1e-5  # RMSprop epsilon
MAX_GRAD_NORM = 0.5  # Maximum gradient norm

# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def main():
    print("Starting main function...")
    STAGE_NAME = "SuperMarioBros-1-1-v3"

    env = get_env(STAGE_NAME, N_ENVS)
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

    save_dir = Path("./model_a2c2")
    save_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = save_dir / "reward_log.csv"

    with open(reward_log_path, "a") as f:
        print("timesteps,reward,best_reward", file=f)

    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,  # Change to 1 to see training progress
        policy_kwargs=dict(
            optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)
        ),
        # policy_kwargs=policy_kwargs,
        tensorboard_log=save_dir,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        rms_prop_eps=RMS_PROP_EPS,
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
