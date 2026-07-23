"""Smoke test: the package imports and its core runtime dep is present.

Exists so pytest and CI have something green to run before the first real
module lands. Delete once `tests/core/` has real coverage.
"""

import tfm_lens


def test_package_has_version():
    assert tfm_lens.__version__


def test_torch_importable():
    import torch

    assert torch.__version__
