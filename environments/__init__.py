# Automatically detect observation and action spaces to configure the agent.

import gymnasium as gym

def create_env(env_name: str):
    try:
        env = gym.make(env_name)
    except gym.error.Error:
        env = gym.make(env_name)
    return env
