"""Shared fixtures for the skeleton test suite.

The toy adapter and a sample input are reused across the adapter / capture /
interventions / logit_lens tests, so they live here rather than in one file.
"""

import os
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from tfm_lens.adapters.base import ModelAdapter
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


class _DoubleStreamBlock(nn.Module):
    """A block that takes two streams (support, query) and returns both — the Mitra
    convention, to exercise the core's double-stream skip."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, support, query):
        return self.lin(support), self.lin(query)


class _DoubleStreamBackbone(nn.Module):
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.blocks = nn.ModuleList(_DoubleStreamBlock(hidden) for _ in range(n_layers))

    def forward(self, support, query):
        for blk in self.blocks:
            support, query = blk(support, query)
        return query


class ToyAdapterDoubleStream(ModelAdapter):
    """3D double-stream adapter (Mitra family): layers take/return (support, query),
    readout takes the query stream. Only the fixture below uses it."""

    HIDDEN = ToyAdapter3D.HIDDEN
    N_LAYERS = ToyAdapter3D.N_LAYERS

    def __init__(self):
        self.backbone = _DoubleStreamBackbone(self.N_LAYERS, self.HIDDEN)

    @property
    def layers(self):
        return list(self.backbone.blocks)

    def forward_frozen(self, X, y_train, eval_pos):
        with torch.no_grad():
            self.backbone(X[:, :eval_pos], X[:, eval_pos:])  # support, query

    def decoder_template(self):
        return nn.Linear(self.HIDDEN, ToyAdapter3D.N_CLASSES)

    def to(self, device):
        self.backbone = self.backbone.to(device)
        return self

    def readout(self, out, eval_pos):
        return out[1]  # query stream (3D, already the test rows)

    def identity_forward(self, *args, **kwargs):
        return args[0], args[1]

    def residual_of(self, layer_output):
        return layer_output[0], layer_output[1]  # both streams (support, query)

    def inject_delta_forward(self, donor_delta, *args, **kwargs):
        return args[0] + donor_delta[0], args[1] + donor_delta[1]


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
def toy_adapter_double_stream() -> ToyAdapterDoubleStream:
    return ToyAdapterDoubleStream()


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
