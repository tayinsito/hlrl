import gymnasium as gym
import torch
from train import PolicyNetwork

def test_model(env_name, model_path):
    env = gym.make(env_name, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy_net = PolicyNetwork(obs_dim, act_dim, hidden_size=128)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = policy_net(obs)
            action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Test completed. Total Reward = {total_reward}")

if __name__ == "__main__":
    test_model("CartPole-v1", "policy_net.pth")

