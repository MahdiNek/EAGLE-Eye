from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from torch.nn import Module, Sequential
import torchvision

class AttrDict(dict):
    """
    Subclass dict and define getter-setter. This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

def convert_dict_to_attrdict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_dict_to_attrdict(v)
            d[k] = v

    if isinstance(d, dict):
        d = AttrDict(d)

    return d


def load_config(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f,Loader=yaml.FullLoader)

        data = AttrDict(data)
    data = convert_dict_to_attrdict(data)

    return data


