# Adapt the policy network based on the action space.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(ContinuousPolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp()
        return mean, std
