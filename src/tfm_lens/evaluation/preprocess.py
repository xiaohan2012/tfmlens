"""Drive the vendored LimiX preprocessing on one real table.

Reproduces the no-retrieval single-worker pipeline that LimiX's predictor builds
(FilterValidFeatures -> quantile rebalance -> ordinal categorical encode ->
feature shuffle) so real OpenML tables land in the distribution the frozen model
was trained on. This is worker ① of ``cls_default_noretrieval.json``; the 4-way
ensemble is deferred.
"""

import numpy as np

from tfm_lens.vendor.limix.preprocess import (
    CategoricalFeatureEncoder,
    FeatureShuffler,
    FilterValidFeatures,
    RebalanceFeatureDistribution,
)

# worker ① of LimiX's cls_default_noretrieval.json
_WORKER = {
    "RebalanceFeatureDistribution": {
        "worker_tags": ["quantile_uniform_10"],
        "discrete_flag": False,
        "original_flag": True,
        "svd_tag": "svd",
    },
    "CategoricalFeatureEncoder": {"encoding_strategy": "ordinal_strict_feature_shuffled"},
    "FeatureShuffler": {"mode": "shuffle"},
}


def limix_preprocess(X_train, y_train, X_test, categorical_idx, seed=0):
    """Preprocess a (train, test) table for the frozen LimiX forward.

    Fits on the train (support) rows — the transforms read ``eval_pos = len(y)``
    — and transforms both. Returns ``(X_train_p, X_test_p)`` float32 arrays ready
    for ``predict_layers``.
    """
    x = np.concatenate([np.asarray(X_train), np.asarray(X_test)], axis=0).astype(np.float32)
    cat = list(categorical_idx)
    y = np.asarray(y_train)

    steps = [
        FilterValidFeatures(),
        RebalanceFeatureDistribution(**_WORKER["RebalanceFeatureDistribution"]),
        CategoricalFeatureEncoder(**_WORKER["CategoricalFeatureEncoder"]),
        FeatureShuffler(**_WORKER["FeatureShuffler"]),
    ]
    for step in steps:
        x, cat = step.fit_transform(x, cat, seed, y=y)

    eval_pos = len(y_train)
    return x[:eval_pos], x[eval_pos:]
