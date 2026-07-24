"""LimiX adapter — the first real-model ModelAdapter.

Wraps a frozen LimiX ``FeaturesTransformer`` (built by the vendored loader) so
capture / interventions / logit_lens work on it unchanged. LimiX is a 4D-residual
model: layer outputs are ``[batch, seq, tokens, hidden]`` with the label token
last on the token axis, and each layer returns a 3-tuple.
"""

import torch

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.vendor.limix import load_model


class LimixAdapter(ModelAdapter):
    needs_transpose = True  # cls_y_decoder consumes (seq, batch, hidden)

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu"):
        return cls(load_model(path)).to(device)  # load_model lands on CPU

    def to(self, device: str) -> "LimixAdapter":
        self.model = self.model.to(device)
        return self

    @property
    def layers(self):
        return list(self.model.transformer_encoder.layers)

    def forward_frozen(self, X, y_train, eval_pos: int) -> None:
        with torch.no_grad():
            self.model(X, y_train, eval_pos, task_type="cls")

    def decoder_template(self):
        return self.model.cls_y_decoder

    def select_label_token(self, emb):
        return emb[:, :, -1, :]  # 4D -> 3D: keep the label token

    def post_norm(self, emb):
        return self.model.encoder_out_norm(emb)

    def identity_forward(self, x):
        return x, None, None  # LimiX layers return (residual, feat_attn, sample_attn)
