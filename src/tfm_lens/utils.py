"""Small shared helpers with no tfm_lens dependencies."""

import random

import numpy as np
import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
