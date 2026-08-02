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
    # Token-axis index of the label token (the only position the decoder reads), for
    # resample ablation's label←label bucketing. None = no token axis (3D residual).
    label_token_index: int | None = None

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
        """A layer's raw output -> ``[batch, n_test, hidden]``, decoder-ready.

        Steps a subclass may need to override:

        - pick the residual stream (unwrap a tuple output; pick a 2nd stream)
        - select the label-bearing token (4D models)
        - apply a pre-decoder norm (LimiX / TabICL)
        - keep only the test rows

        Default: single 3D stream, no norm — take the residual, slice test rows.
        """
        h = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return h[:, eval_pos:]

    # ---- intervention hook (overridable; default suits single-output layers) ----
    def identity_forward(self, *args, **kwargs):
        """What a skipped layer returns, given the layer's inputs. Default: pass the
        first input (the residual) through. Override for tuple-returning layers:

        - LimiX: ``return args[0], None, None``
        - Mitra: ``return args[0], args[1]`` (support, query)
        """
        return args[0] if args else next(iter(kwargs.values()))

    # ---- δ-injection hooks (overridable; default suits single-stream layers) ----
    def residual_of(self, layer_output):
        """The residual stream(s) a layer carries forward, in *raw layer-output*
        coordinates — stream picked, but **no token-slice and no norm** (unlike
        ``readout``, which prepares the decoder input). This is the coordinate the
        contribution ``δ = residual_of(out) − residual_of(in)`` is captured and a
        donor ``δ`` is applied in.

        Default: single stream — unwrap a tuple (LimiX's ``(res, …)``, or the
        capture cache's input tuple ``(x,)``), else the tensor itself. Override for
        double-stream models (Mitra returns both ``(support, query)``).
        """
        return layer_output[0] if isinstance(layer_output, tuple) else layer_output

    def inject_delta_forward(self, donor_delta, *args, **kwargs):
        """What a δ-injected layer returns: the input residual **plus** ``donor_delta``,
        in the layer's output shape. Mirrors ``identity_forward`` (same tuple layout)
        but adds the donor contribution instead of passing the input through.

        Default: single stream — ``input + donor_delta``. Override for tuple-returning
        layers:

        - LimiX: ``return args[0] + donor_delta, None, None``
        - Mitra: ``return args[0] + donor_delta[0], args[1] + donor_delta[1]``
        """
        x = args[0] if args else next(iter(kwargs.values()))
        return x + donor_delta

    @property
    def n_layers(self) -> int:
        return len(self.layers)
