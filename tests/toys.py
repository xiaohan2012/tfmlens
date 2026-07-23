"""Toy backbone + adapter — the stand-in "real model" for all skeleton tests.

A few ``nn.Linear`` "layers" are enough to exercise every seam (capture hooks,
forward-swap interventions, per-layer decoding) with no GPU and no LimiX
checkpoint. capture / interventions / logit_lens tests all reuse ``ToyAdapter``.
"""

import torch
import torch.nn as nn

from tfm_lens.adapters.base import ModelAdapter


class ToyBackbone(nn.Module):
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


class ToyAdapter(ModelAdapter):
    """Minimal ModelAdapter over ToyBackbone."""

    label_token_index = -1
    needs_transpose = False

    HIDDEN = 8
    N_CLASSES = 4
    N_LAYERS = 3

    def __init__(self):
        self.backbone = ToyBackbone(n_layers=self.N_LAYERS, hidden=self.HIDDEN)

    @property
    def layers(self):
        return list(self.backbone.blocks)

    def forward_frozen(self, X, y_train, eval_pos):
        with torch.no_grad():
            self.backbone(X)

    def decoder_template(self):
        return nn.Linear(self.HIDDEN, self.N_CLASSES)
