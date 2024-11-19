import torch
import torch.nn.functional as F
from torch import nn

from ppo.ppo_network import PPONetwork


class Mario_PPO:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # PPO hyperparameters
        self.gamma = 0.9
        self.gae_lambda = 1.0
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.25
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 16

        # Initialize network and optimizer
        self.network = PPONetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.00025)

        # Initialize memory for PPO
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def act(self, state):
        """
        Given a state, choose an action based on the policy network

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        """
        # Preprocess the state
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, value = self.network(state)

        # Sample action from the probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        # Store data for training
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_dist.log_prob(action))
        self.values.append(value)

        return action.item()

    def learn(self):

        # Convert lists to tensors
        states = torch.cat(self.states)
        actions = torch.tensor(self.actions).to(self.device)
        old_action_probs = torch.stack(self.action_probs).to(self.device)
        values = torch.cat(self.values)

        # Compute returns and advantages
        returns = self.compute_returns().unsqueeze(
            1
        )  # Add dimension to match value predictions
        advantages = returns - values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0

        # PPO update
        for _ in range(len(self.rewards) // self.batch_size):
            # Create random indices for batching
            batch_indices = torch.randperm(states.size(0))

            # Process mini-batches
            for start_idx in range(0, states.size(0), self.batch_size):
                # Get batch indices
                batch_idx = batch_indices[start_idx : start_idx + self.batch_size]

                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_action_probs = old_action_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Get new action probabilities and values
                new_action_probs, new_values = self.network(batch_states)
                new_action_dist = torch.distributions.Categorical(new_action_probs)
                new_action_log_probs = new_action_dist.log_prob(batch_actions)

                # Compute ratio and clipped ratio
                ratio = torch.exp(new_action_log_probs - batch_old_action_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )

                # Compute losses
                policy_loss = -torch.min(
                    ratio * batch_advantages.squeeze(),
                    clipped_ratio * batch_advantages.squeeze(),
                ).mean()
                # Add a minimum loss threshold to prevent collapse
                # policy_loss = torch.max(policy_loss, torch.tensor(1e-4).to(self.device))

                # Value loss
                value_loss = F.smooth_l1_loss(new_values, batch_returns)
                # value_loss = torch.max(
                #     value_loss, torch.tensor(1e-4).to(self.device)
                # )  # Clamp value loss

                # Entropy bonus for exploration
                entropy_loss = -new_action_dist.entropy().mean()
                # if (
                #     entropy_loss.item() > -0.1
                # ):  # You might need to adjust this threshold
                #     entropy_loss *= 2

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Store losses for logging
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        # Calculate average losses
        num_updates = max((states.size(0) // self.batch_size) * self.ppo_epochs, 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates

        # Clear memory
        self.clear_memory()

        return avg_policy_loss, avg_value_loss

    def compute_returns(self):
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        values = torch.cat(self.values)

        # Get the next value for non-terminal states
        with torch.no_grad():
            _, next_value = self.network(self.states[-1])
            next_value = next_value.squeeze()

        returns = []
        gae = 0
        next_val = next_value if not self.dones[-1] else 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = next_val
            else:
                next_value = values[step + 1]

            delta = (
                rewards[step]
                + self.gamma * next_value * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        returns = torch.tensor(returns).to(self.device)
        return returns

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def save(self, i):
        save_path = self.save_dir / f"mario_net_{i}.chkpt"
        torch.save(
            dict(model=self.network.state_dict()),
            save_path,
        )
