"""Shared fixtures for the skeleton test suite.

The toy adapter and a sample input are reused across the adapter / capture /
interventions / logit_lens tests, so they live here rather than in one file.
"""

import os
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from toys import ToyAdapter3D, ToyAdapter4D


class _KeywordCallBlock(nn.Module):
    """A Linear block that takes its input by name (``q``) so a backbone can call it
    as ``blk(q=x)``."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, q):
        return self.lin(q)


class _KeywordCallBackbone(nn.Module):
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.blocks = nn.ModuleList(_KeywordCallBlock(hidden) for _ in range(n_layers))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(q=x)  # keyword call, like TabICL's ICL Encoder
        return x


class ToyAdapter3DKeywordCall(ToyAdapter3D):
    """ToyAdapter3D whose backbone drives blocks by keyword (``blk(q=x)``) — TabICL's
    ICL Encoder convention, to exercise the core's kwarg handling. Only the fixture
    below uses it, so it lives here rather than in the shared toys.py."""

    def __init__(self):
        self.backbone = _KeywordCallBackbone(self.N_LAYERS, self.HIDDEN)


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
def toy_adapter() -> ToyAdapter3D:
    return ToyAdapter3D()


@pytest.fixture
def toy_adapter_keyword_call() -> ToyAdapter3DKeywordCall:
    return ToyAdapter3DKeywordCall()


@pytest.fixture
def toy_input() -> torch.Tensor:
    return torch.randn(2, 5, ToyAdapter3D.HIDDEN)  # (batch, seq, hidden)


@pytest.fixture
def toy_decoders(toy_adapter: ToyAdapter3D) -> list:
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
