
from model import Model

from yaml import safe_load


def load_config():
    with open('C:/Users/burov/Documents/pochta/pochta/pochta/config/config.yml', 'r', encoding='utf-8') as f:
        data = safe_load(f)
    return data

data = load_config()

tree = Model(data)
tree.make_tree()
