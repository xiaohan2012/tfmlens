"""Shared fixtures for the skeleton test suite.

The toy adapter and a sample input are reused across the adapter / capture /
interventions / logit_lens tests, so they live here rather than in one file.
"""

import os
from copy import deepcopy

import pytest
import torch

from toys import ToyAdapter, ToyAdapter4D


@pytest.fixture(scope="session")
def limix_ckpt() -> str:
    """Path to LimiX-2M.ckpt: use $LIMIX_2M_CKPT if set, else fetch from HF."""
    path = os.environ.get("LIMIX_2M_CKPT")
    if path and os.path.exists(path):
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pytest.skip("huggingface_hub not installed and LIMIX_2M_CKPT not set")
    try:
        return hf_hub_download(repo_id="stableai-org/LimiX-2M", filename="LimiX-2M.ckpt")
    except Exception as exc:  # offline / HF unavailable
        pytest.skip(f"could not fetch LimiX-2M ckpt: {exc}")


@pytest.fixture(scope="session")
def limix_model(limix_ckpt: str):
    from tfm_lens.vendor.limix import load_model

    return load_model(limix_ckpt)


@pytest.fixture
def toy_adapter() -> ToyAdapter:
    return ToyAdapter()


@pytest.fixture
def toy_input() -> torch.Tensor:
    return torch.randn(2, 5, ToyAdapter.HIDDEN)  # (batch, seq, hidden)


@pytest.fixture
def toy_decoders(toy_adapter: ToyAdapter) -> list:
    # one decoder per capture depth: n_layers + 1 (see capture cache length).
    return [deepcopy(toy_adapter.decoder_template()) for _ in range(toy_adapter.n_layers + 1)]


@pytest.fixture
def toy_adapter_4d() -> ToyAdapter4D:
    return ToyAdapter4D()


@pytest.fixture
def toy_input_4d() -> torch.Tensor:
    # (batch, seq, tokens, hidden)
    return torch.randn(2, 5, ToyAdapter4D.TOKENS, ToyAdapter4D.HIDDEN)


@pytest.fixture
def toy_decoders_4d(toy_adapter_4d: ToyAdapter4D) -> list:
    return [deepcopy(toy_adapter_4d.decoder_template()) for _ in range(toy_adapter_4d.n_layers + 1)]
