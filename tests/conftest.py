"""Shared fixtures for the skeleton test suite.

The toy adapter and a sample input are reused across the adapter / capture /
interventions / logit_lens tests, so they live here rather than in one file.
"""

import pytest
import torch

from toys import ToyAdapter


@pytest.fixture
def toy_adapter() -> ToyAdapter:
    return ToyAdapter()


@pytest.fixture
def toy_input() -> torch.Tensor:
    return torch.randn(2, 5, ToyAdapter.HIDDEN)  # (batch, seq, hidden)
