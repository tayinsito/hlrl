# Import config.yaml

import yaml

def load_config(env_name: str, config_path="utils/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get(env_name, {})
