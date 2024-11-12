import random
import numpy as np
import torch
import os

def seed_worker(worker_id):
    """
    for dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    print(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=0,fast=False):
    """
    for main thread
    cudnn default setting: benchmark False deterministic False enabled True
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    # gather etc. module will give deterministic result
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except:
    #     pass

    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Remove randomness (may be slower on Tesla GPUs)
        # https://pytorch.org/docs/stable/notes/randomness.html
        if not fast:
            torch.backends.cudnn.enabled = False # not use cudnn
            torch.backends.cudnn.deterministic = True # using cudnn deterministic algorithm
        else:
            # use cudnn and use benchmark to enhance efficiency
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        # The model input data size is fixed, using benchmark to enhance efficiency
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.benchmark = True
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

