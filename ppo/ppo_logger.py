import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanPolicyLoss':>15}{'MeanValueLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_policy_losses_plot = save_dir / "policy_loss_plot.jpg"
        self.ep_value_losses_plot = save_dir / "value_loss_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_policy_losses = []
        self.ep_value_losses = []

        # Moving averages
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_policy_losses = []
        self.moving_avg_ep_value_losses = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, policy_loss=None, value_loss=None, length=None):
        self.curr_ep_reward = reward
        self.curr_ep_length = length  # Set the actual episode length
        if policy_loss is not None:
            self.curr_ep_policy_loss = policy_loss
            self.curr_ep_value_loss = value_loss
            self.curr_ep_loss_length = 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(
            self.curr_ep_length
        )  # This will now store the correct length
        self.ep_policy_losses.append(self.curr_ep_policy_loss)
        self.ep_value_losses.append(self.curr_ep_value_loss)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_policy_loss = 0.0
        self.curr_ep_value_loss = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_policy_loss = np.round(np.mean(self.ep_policy_losses[-100:]), 3)
        mean_ep_value_loss = np.round(np.mean(self.ep_value_losses[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_policy_losses.append(mean_ep_policy_loss)
        self.moving_avg_ep_value_losses.append(mean_ep_value_loss)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Policy Loss {mean_ep_policy_loss} - "
            f"Mean Value Loss {mean_ep_value_loss} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_policy_loss:15.3f}{mean_ep_value_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in [
            "ep_lengths",
            "ep_policy_losses",
            "ep_value_losses",
            "ep_rewards",
        ]:
            plt.clf()
            plt.plot(
                getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}"
            )
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))
