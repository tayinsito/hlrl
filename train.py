import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# Training Function
def train(env, policy_net, config):
    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    gamma = config["gamma"]
    max_episodes = config["max_episodes"]

    for episode in range(max_episodes):
        obs, _ = env.reset()  # Gymnasium reset returns obs and additional info
        obs = torch.tensor(obs, dtype=torch.float32)

        log_probs, rewards, values = [], [], []
        done = False
        while not done:
            logits, value = policy_net(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            log_prob = probs.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            next_obs = torch.tensor(next_obs, dtype=torch.float32)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            obs = next_obs
            done = terminated or truncated  # Adjust for Gymnasium's done logic

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Compute advantages
        values = torch.stack(values).squeeze()
        advantages = returns - values

        # Update policy and value network
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{max_episodes}: Total Reward = {sum(rewards)}")

# Main Function
def main(args):
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy_net = PolicyNetwork(obs_dim, act_dim, hidden_size=128)

    config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "max_episodes": 1000,
    }

    train(env, policy_net, config)
    torch.save(policy_net.state_dict(), "policy_net.pth")
    print("Training complete. Model saved as 'policy_net.pth'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    args = parser.parse_args()

    main(args)

