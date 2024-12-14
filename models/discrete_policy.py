# Adapt the policy network based on the action space.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(DiscretePolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits
