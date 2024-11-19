from torch import nn

# Gym is an OpenAI toolkit for RL

# NES Emulator for OpenAI Gym

# Super Mario environment for OpenAI Gym


class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy = nn.Sequential(nn.Linear(512, output_dim), nn.Softmax(dim=-1))

        # Value head (critic)
        self.value = nn.Sequential(nn.Linear(512, 1))

    def forward(self, x):
        features = self.features(x)
        action_probs = self.policy(features)
        value = self.value(features)
        return action_probs, value
