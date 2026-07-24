"""The ModelAdapter contract.

A pure contract: it holds no model knowledge and no mechanism (hooks, skips,
decoding live elsewhere). To plug a frozen tabular FM into tfm_lens, subclass
this and answer three questions — where are your layers, how do you run a
forward, and what does your decoder head look like — plus a couple of optional
capability declarations. capture / interventions / logit_lens then talk only to
this interface, never to the concrete model.
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class ModelAdapter(ABC):
    """Adapts a frozen backbone into a per-layer-readable, intervenable object."""

    # Capability declaration — static, overridden by subclasses as needed.
    needs_transpose: bool = False  # whether the decoder wants (seq, batch, hidden)

    # ---- must be implemented ----
    @property
    @abstractmethod
    def layers(self) -> list[nn.Module]:
        """The layer modules to hook / whose forward can be swapped."""

    @abstractmethod
    def forward_frozen(self, X, y_train, eval_pos: int) -> None:
        """Run one forward under no_grad, only to trigger hooks; returns nothing."""

    @abstractmethod
    def decoder_template(self) -> nn.Module:
        """The backbone's decoder head, deepcopied into per-layer decoders."""

    # ---- decode-path hooks (overridable; defaults suit 3D models) ----
    def select_label_token(self, emb):
        """Reduce a raw layer residual to [batch, seq, hidden].

        3D models (TabPFN v1 / TabICL / the toy) are already [batch, seq, hidden]
        -> identity. 4D models (LimiX / TabPFN v2) hold the label token on the
        token axis and override this, e.g. ``return emb[:, :, -1, :]``.
        """
        return emb

    def post_norm(self, emb):
        """Normalization applied before the decoder; identity by default."""
        return emb

    # ---- intervention hook (overridable; default suits single-output layers) ----
    def identity_forward(self, x):
        """What a skipped layer should return. Default passes ``x`` through; models
        whose layers return a tuple override, e.g. ``return (x, None, None)``.
        """
        return x

    @property
    def n_layers(self) -> int:
        return len(self.layers)
