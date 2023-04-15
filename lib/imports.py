import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
import functools
import os
import numpy as np

import warnings
import glob


import io
from joblib import delayed 

from cacher import root, file_cached, mem_cached, clear_cache

import pymorphy3

from corus import load_lenta2
from navec import Navec
from razdel import tokenize, sentenize
from nerus import load_nerus

from utils import ProgressParallel, chunks, size_of_tensor, count_parameters
from joblib import delayed

from utils import download_file

from slovnet.model.emb import NavecEmbedding

# from torchmetrics.functional.classification import binary_accuracy

# import nest_asyncio
# nest_asyncio.apply()


import random
import string
import importlib
from pymorphy3.tagset import OpencorporaTag
from params import NO_PUNCT, build_params
import dill


import socket
import dill
import torch
import concurrent
import io
import queue
import threading

