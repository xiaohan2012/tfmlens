"""Mitra adapter — the vendored Tab2D as a ModelAdapter.

Tab2D is a double-stream 4D model: each layer processes support (context) and
query (test) separately and returns ``(support, query)``. the ablation sweep reads the
query stream. Loaded with the standard-attention path forced on (no flash), so
the adapter maintains a single forward / residual layout.

Not registered in adapters/__init__ — it imports the optional ``mitra`` group,
so a lighter install never touches it. Import it explicitly:
``from tfm_lens.adapters.mitra import MitraAdapter``.
"""

import torch

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.vendor.mitra.tab2d import Tab2D

_HF_REPO = "autogluon/mitra-classifier"


class MitraAdapter(ModelAdapter):
    label_token_index = 0  # label token is first on the token axis (einops.pack((y, x)))

    def __init__(self, model):
        self.model = model
        # Force the standard-attention path (see module doc): the flash path has a
        # different layer signature and residual format we don't adapt to.
        for m in model.modules():
            if hasattr(m, "use_flash_attn"):
                m.use_flash_attn = False

    @classmethod
    def from_checkpoint(cls, path: str | None = None, device: str = "cpu") -> "MitraAdapter":
        """Load Tab2D from a checkpoint; ``path=None`` downloads the classifier
        (``autogluon/mitra-classifier``) from HF (config.json + model.safetensors)."""
        model = Tab2D.from_pretrained(path or _HF_REPO, device=device)
        model.eval()
        return cls(model).to(device)

    def to(self, device: str) -> "MitraAdapter":
        self.model = self.model.to(device)
        return self

    @property
    def layers(self):
        return list(self.model.layers)

    def forward_frozen(self, X, y_train, eval_pos: int) -> None:
        # Split the packed table into Mitra's support (train) / query (test) streams
        # and pass all-False padding (no missing rows/features), then run the forward.
        x_support, x_query = X[:, :eval_pos], X[:, eval_pos:]
        b, n_s, f = x_support.shape
        n_q = x_query.shape[1]
        pad_f = torch.zeros(b, f, dtype=torch.bool, device=X.device)
        pad_s = torch.zeros(b, n_s, dtype=torch.bool, device=X.device)
        pad_q = torch.zeros(b, n_q, dtype=torch.bool, device=X.device)
        with torch.no_grad():
            self.model(x_support, y_train, x_query, pad_f, pad_s, pad_q)

    def decoder_template(self):
        return self.model.final_layer

    def readout(self, out, eval_pos: int):
        # Mitra decode path: double stream -> query (index 1); 4D residual -> label
        # token (index 0) -> final norm. query is already all test rows, so no slice.
        h = out[1][:, :, 0, :]
        return self.model.final_layer_norm(h)

    def identity_forward(self, *args, **kwargs):
        # A skipped layer returns its two streams unchanged: (support, query).
        return args[0], args[1]

    def residual_of(self, layer_output):
        # Double-stream: both (support, query) carry forward -> δ is captured/applied
        # on both. layer_output is the 2-tuple (from a layer output or the capture
        # cache's input tuple).
        return layer_output[0], layer_output[1]

    def inject_delta_forward(self, donor_delta, *args, **kwargs):
        # donor_delta is a (support_δ, query_δ) pair, one per stream.
        return args[0] + donor_delta[0], args[1] + donor_delta[1]
