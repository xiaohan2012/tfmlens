"""Toy backbone + adapter — the stand-in "real model" for all skeleton tests.

A few ``nn.Linear`` "layers" are enough to exercise every seam (capture hooks,
forward-swap interventions, per-layer decoding) with no GPU and no LimiX
checkpoint. ``ToyAdapter3D`` covers the 3D residual family (TabPFN-v1 / TabICL);
``ToyAdapter4D`` covers the 4D family (LimiX / TabPFN-v2).
"""

import torch
import torch.nn as nn

from tfm_lens.adapters.base import ModelAdapter


class ToyBackbone3D(nn.Module):
    """N independent Linear "layers" applied in sequence.

    Each block is its own ``nn.Module`` so a forward hook can read its output
    and an intervention can swap its ``.forward``.
    """

    def __init__(self, n_layers: int = 3, hidden: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))

    def forward(self, x):  # x: (batch, seq, hidden)
        for blk in self.blocks:
            x = blk(x)
        return x


class ToyAdapter3D(ModelAdapter):
    """Minimal 3D ModelAdapter over ToyBackbone3D (TabPFN-v1 / TabICL family)."""

    needs_transpose = False

    HIDDEN = 8
    N_CLASSES = 4
    N_LAYERS = 3

    def __init__(self):
        self.backbone = ToyBackbone3D(n_layers=self.N_LAYERS, hidden=self.HIDDEN)

    @property
    def layers(self):
        return list(self.backbone.blocks)

    def forward_frozen(self, X, y_train, eval_pos):
        with torch.no_grad():
            self.backbone(X)

    def decoder_template(self):
        return nn.Linear(self.HIDDEN, self.N_CLASSES)


class ToyBlock4D(nn.Module):
    """A 4D layer: operates on [batch, seq, tokens, hidden] and returns a 3-tuple,
    mimicking LimiX layers (residual, feature_attn, sample_attn)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x):  # x: (batch, seq, tokens, hidden)
        return self.lin(x), None, None


class ToyBackbone4D(nn.Module):
    def __init__(self, n_layers: int = 3, hidden: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(ToyBlock4D(hidden) for _ in range(n_layers))

    def forward(self, x):
        for blk in self.blocks:
            x, _, _ = blk(x)
        return x


class ToyAdapter4D(ModelAdapter):
    """A 4D ModelAdapter (LimiX / TabPFN-v2 family): label token on the token axis,
    layers return a 3-tuple. Exercises the 4D decode path with no checkpoint."""

    needs_transpose = False

    HIDDEN = 8
    N_CLASSES = 4
    N_LAYERS = 3
    TOKENS = 4  # e.g. feature-group tokens + 1 label token

    def __init__(self):
        self.backbone = ToyBackbone4D(n_layers=self.N_LAYERS, hidden=self.HIDDEN)

    @property
    def layers(self):
        return list(self.backbone.blocks)

    def forward_frozen(self, X, y_train, eval_pos):
        with torch.no_grad():
            self.backbone(X)

    def decoder_template(self):
        return nn.Linear(self.HIDDEN, self.N_CLASSES)

    def select_label_token(self, emb):
        return emb[:, :, -1, :]  # 4D -> 3D: keep the label token

    def identity_forward(self, x):
        return x, None, None  # match the layer's 3-tuple return
