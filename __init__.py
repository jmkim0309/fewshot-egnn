import numpy as np
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch import utils
from torch.nn import functional as F
from torch.utils.data import *
from torch.distributions import *
from torchtools import tt


__author__ = 'namju.kim@kakaobrain.com'


# initialize seed
if tt.arg.seed:
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
