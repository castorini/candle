import argparse

class Configs(object):
    @staticmethod
    def base_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--prune_activation", type=str, default=None, choices=[None, "thresh_tanh"])
        parser.add_argument("--prune_method", type=str, default="magnitude", choices=["magnitude"])
        parser.add_argument("--use_cpu", action="store_true", default=False)
        return parser.parse_known_args()[0]

def read_config(name="base"):
    return _configs[name]()

def register_config_reader(name, reader):
    _configs[name] = reader

_configs = dict(base=Configs.base_config)