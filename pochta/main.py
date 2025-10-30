
from model import Model

from yaml import safe_load


def load_config():
    with open('config/config.py', 'r') as f:
        data = safe_load(f)
    return data


data = load_config()

retriver = Model(data)
