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

    @abstractmethod
    def to(self, device: str) -> "ModelAdapter":
        """Move the underlying backbone to ``device``; return self (chainable)."""

    # ---- decode-path hook (overridable) ----
    def readout(self, layer_output, eval_pos: int):
        """One layer's raw captured output -> ``[batch, n_test, hidden]``, ready for
        a decoder. Concretely: pick the residual stream, select the token that
        carries the label, apply any pre-decoder norm, and keep only the test rows.

        The default suits a single-stream model with no pre-decoder norm whose
        residual is already ``[batch, seq, hidden]`` (the 3D toy / TabPFN-v1 family):
        take the residual (unwrapping a tuple output) and slice off the support rows.
        Override to add token selection (4D), a norm (TabICL/LimiX), or to pick a
        second stream (Mitra's query).
        """
        h = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return h[:, eval_pos:]

    # ---- intervention hook (overridable; default suits single-output layers) ----
    def identity_forward(self, x):
        """What a skipped layer should return. Default passes ``x`` through; models
        whose layers return a tuple override, e.g. ``return (x, None, None)``.
        """
        return x

    @property
    def n_layers(self) -> int:
        return len(self.layers)
