"""Vendored TabICL prior — synthetic tabular data generation (Apache/BSD-3).

See NOTICE for provenance. Only the code needed to generate on-the-fly training
tables (the ``mix_scm`` prior) is used; ``PriorDataset`` is the on-the-fly entry
point (an ``IterableDataset`` yielding batches of synthetic tables). tfm_lens's
data/prior.py wraps this; it does not import upstream tabicl.
"""

from .dataset import PriorDataset

__all__ = ["PriorDataset"]
