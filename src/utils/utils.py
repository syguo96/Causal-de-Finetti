import random
import os
import numpy as np
import torch
from typing import Iterable

def set_seed(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def initialize_results(results, methods):
    for key in results.keys():
        for method in methods:
            results[key][method] = []

def initialize_records_per_env(methods):
    counts_per_env = {}
    time_per_env = {}
    for method in methods:
        counts_per_env[method] = []
        time_per_env[method]= []

    return counts_per_env, time_per_env

def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x