"""Full-system reproducibility and deterministic random seeding utilities.

Ensures reproducibility across Python random, NumPy, PyTorch CPU & CUDA,
and OS hashing to guarantee reproducible evaluation.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Generator

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Fix random seeds across all standard libraries and frameworks.

    Parameters
    ----------
    seed : int
        Deterministic random seed (default 42).
    deterministic_cudnn : bool
        If True and CUDA is available, forces deterministic algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_seed(seed: int = 42) -> None:
    """Convenience alias for :func:`seed_everything`."""
    seed_everything(seed)


@contextlib.contextmanager
def temp_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily run code within a localized deterministic random seed scope.

    Restores the previous Python, NumPy, and PyTorch random states upon exit.
    """
    # Save states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

    try:
        seed_everything(seed)
        yield
    finally:
        # Restore states
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_cpu_state)
        if torch.cuda.is_available():
            for i, state in enumerate(cuda_states):
                torch.cuda.set_rng_state(state, i)
