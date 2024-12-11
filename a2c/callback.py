import os
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):
    def __init__(
        self,
        check_freq,
        save_path,
        episode_numbers,
        env,
        max_timestep_test,
        model,
        total_timesteps,
        reward_log_path,
    ):
        super(TrainAndLoggingCallback, self).__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.episode_numbers = episode_numbers
        self.env = env
        self.max_timestep_test = max_timestep_test
        self.model = model
        self.total_timesteps = total_timesteps
        self.reward_log_path = reward_log_path
        self.best_mean_reward = -float("inf")

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = f"{self.save_path}/best_model_{self.n_calls}.zip"
            self.model.save(model_path)

            mean_reward = 0
            n_episodes = 0

            for _ in range(self.episode_numbers):
                obs = self.env.reset()
                total_time = [0] * self.env.num_envs
                episode_rewards = [0] * self.env.num_envs
                dones = [False] * self.env.num_envs

                while not all(dones) and all(
                    t < self.max_timestep_test for t in total_time
                ):
                    action, _ = self.model.predict(obs)
                    obs, rewards, new_dones, _ = self.env.step(action)

                    for i in range(self.env.num_envs):
                        if not dones[i]:
                            total_time[i] += 1
                            episode_rewards[i] += rewards[i]
                            dones[i] = new_dones[i]

                mean_reward += sum(episode_rewards)
                n_episodes += self.env.num_envs

            mean_reward = mean_reward / n_episodes

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"{self.save_path}/best_model.zip")

            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print(f"Last mean reward per episode: {mean_reward:.2f}")

            # Log the rewards
            with open(self.reward_log_path, "a") as f:
                print(
                    f"{self.num_timesteps},{mean_reward:.2f},{self.best_mean_reward:.2f}",
                    file=f,
                )

        return True
