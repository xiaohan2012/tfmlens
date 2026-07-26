"""TabICL adapter — TabICLv2 as a ModelAdapter.

Wraps the raw TabICL module (three sequential stages: column-wise embedding ->
row-wise interaction -> dataset-wise in-context learning). The logit-lens target
is the ICL stage (``icl_predictor``): its transformer blocks are the "layers",
and ``icl_predictor.decoder`` / ``.ln`` are the decode head / final norm. Labels
are injected into the train rows inside the ICL stage, so the ICL residuals we
capture already see the in-context supervision, matching LimiX.

TabICL is a 3D-residual model — layer outputs are ``[batch, rows, hidden]`` with
the label information distributed across rows, so token selection is identity and
the decoder consumes ``[batch, rows, hidden]`` directly (no transpose).

Not registered in ``adapters/__init__`` on purpose: it imports the optional
``tabicl`` package (the ``tabicl`` dependency group), so a LimiX-only install
never touches it. Import it explicitly: ``from tfm_lens.adapters.tabicl import
TabICLAdapter``.
"""

import torch
from huggingface_hub import hf_hub_download
from tabicl._model.tabicl import TabICL

from tfm_lens.adapters.base import ModelAdapter

_HF_REPO = "jingang/TabICL"
_V2_CKPT = "tabicl-classifier-v2-20260212.ckpt"  # the TabICLv2-paper checkpoint


class TabICLAdapter(ModelAdapter):
    needs_transpose = False  # icl_predictor.decoder consumes [batch, rows, hidden]

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_checkpoint(cls, path: str | None = None, device: str = "cpu") -> "TabICLAdapter":
        """Build the raw TabICL module from a checkpoint (config + state_dict).

        ``path=None`` downloads the TabICLv2 checkpoint from Hugging Face
        (``jingang/TabICL``); pass a local ``.ckpt`` to use it directly. Bypasses
        the sklearn wrapper's ensembling/preprocessing — we want the raw module.
        """
        if path is None:
            path = hf_hub_download(repo_id=_HF_REPO, filename=_V2_CKPT)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = TabICL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model).to(device)

    def to(self, device: str) -> "TabICLAdapter":
        self.model = self.model.to(device)
        return self

    @property
    def layers(self):
        return list(self.model.icl_predictor.tf_icl.blocks)

    def forward_frozen(self, X, y_train, eval_pos: int) -> None:
        # _train_forward runs col -> row -> icl; the ICL stage (ICLearning.forward)
        # is gated on module.training: train mode -> _icl_predictions (inject y ->
        # tf_icl -> ln -> decoder, which fires our block hooks), eval mode -> the
        # inference predictor (KV cache / hierarchical, with a "same #classes per
        # table" assert we don't want). Force train mode for the pass — dropout is 0
        # so there's no numeric effect — and restore the prior mode after.
        was_training = self.model.training
        self.model.train()
        try:
            with torch.no_grad():
                self.model._train_forward(X, y_train)
        finally:
            self.model.train(was_training)

    def decoder_template(self):
        return self.model.icl_predictor.decoder

    def readout(self, out, eval_pos: int):
        h = out[0] if isinstance(out, tuple) else out  # single tensor (3D residual)
        h = self.model.icl_predictor.ln(h)  # pre-norm arch: ln sits before the decoder
        return h[:, eval_pos:]  # keep only the test rows

    # identity_forward: return x — base default (ICL blocks return a single Tensor)
