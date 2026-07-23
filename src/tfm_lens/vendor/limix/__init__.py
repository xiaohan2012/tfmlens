"""Vendored LimiX model definition (Apache-2.0). See NOTICE for provenance.

Only the code needed to construct the frozen LimiX backbone and run a forward is
included (transformer + layer + encoders + a loader). The tfm_lens LimixAdapter
wraps a model built by ``loading.load_model``; it does not import upstream LimiX.
"""

from .loading import load_model

__all__ = ["load_model"]
