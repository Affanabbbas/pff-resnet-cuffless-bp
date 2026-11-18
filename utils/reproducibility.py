import os
import random
import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Set all major random seeds for reproducible training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # warn_only=True avoids crashing if some CUDA op has no deterministic version
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    """
    Make DataLoader workers reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    """
    Generator used by DataLoader, especially for shuffle=True.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
