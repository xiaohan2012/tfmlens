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
    label_token_index = -1  # label token is last on the token axis (see transformer.py)

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_checkpoint(cls, path: str | None = None, device: str = "cpu") -> "LimixAdapter":
        """Load LimiX from a checkpoint; ``path=None`` downloads LimiX-2M from HF
        (mirrors TabICLAdapter.from_checkpoint so the finetune factory is symmetric)."""
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id="stableai-org/LimiX-2M", filename="LimiX-2M.ckpt")
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

    def readout(self, out, eval_pos: int):
        """LimiX decode path (see ModelAdapter.readout): 4D residual -> label token
        (last on the token axis) -> encoder_out_norm -> test rows."""
        h = out[0] if isinstance(out, tuple) else out  # layer output -> residual stream
        h = h[:, :, -1, :]  # 4D -> 3D: keep the label token (last on the token axis)
        h = self.model.encoder_out_norm(h)
        return h[:, eval_pos:]  # keep only the test rows

    def identity_forward(self, *args, **kwargs):
        return args[0], None, None  # LimiX layers return (residual, feat_attn, sample_attn)

    # residual_of: base default (out[0]) already picks the residual from the 3-tuple.
    def resample_forward(self, donor_delta, *args, **kwargs):
        return args[0] + donor_delta, None, None  # match the layer's 3-tuple return
