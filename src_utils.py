import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Logging
logger = logging.getLogger(__name__)

# Helper functions
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def save_json(fname, data):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)

def get_device(cuda_id=None):
    if cuda_id is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cuda_id))
    return device

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp