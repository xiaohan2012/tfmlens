"""TabFM adapter — the vendored PyTorch TabFM as a ModelAdapter.

TabFM is single-stream 3D, like TabICL: ``forward`` runs the col/row embedding
stages then the ICL transformer (``icl_predictor``); self-repair reads the ICL
stage. Everything maps onto the base contract with no core changes:

- ``layers`` -> ``icl_predictor.tf_icl.blocks``
- ``readout`` -> ``icl_predictor.ln`` (RMSNorm) then slice the test rows
- ``decoder_template`` -> ``icl_predictor.decoder`` (MLP)

Not registered in adapters/__init__ — it imports the optional ``tabfm`` group.
"""

import torch

from tfm_lens.adapters.base import ModelAdapter


class TabFMAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_checkpoint(cls, path: str | None = None, device: str = "cpu") -> "TabFMAdapter":
        """Load TabFM (google/tabfm-1.0.0-pytorch, classification) via its HF loader.

        ``path=None`` downloads from HF; ``dtype=None`` keeps float32 (TabFM defaults
        to bf16, but the per-depth decoders and AUC want fp32).
        """
        from tfm_lens.vendor.tabfm.loader import load

        model = load(model_type="classification", checkpoint_path=path, device=device, dtype=None)
        return cls(model)  # load already puts it on device + eval

    def to(self, device: str) -> "TabFMAdapter":
        self.model = self.model.to(device)
        return self

    @property
    def layers(self):
        return list(self.model.icl_predictor.tf_icl.blocks)

    def forward_frozen(self, X, y_train, eval_pos: int) -> None:
        # TabFM.forward wants y over ALL rows (train labels injected via a train_size
        # mask, test rows ignored) plus a per-batch train_size. Pad y with dummies.
        b, seq, _ = X.shape
        dummy = torch.zeros(b, seq - eval_pos, dtype=y_train.dtype, device=X.device)
        y = torch.cat([y_train, dummy], dim=1)
        train_size = torch.full((b,), eval_pos, dtype=torch.long, device=X.device)
        with torch.no_grad():
            self.model(X, y, train_size)  # col -> row -> icl; hooks fire on ICL blocks

    def decoder_template(self):
        return self.model.icl_predictor.decoder

    def readout(self, out, eval_pos: int):
        h = out[0] if isinstance(out, tuple) else out  # ICL block output, 3D [B, T, E]
        h = self.model.icl_predictor.ln(h)  # RMSNorm sits before the decoder
        return h[:, eval_pos:]  # keep only the test rows
