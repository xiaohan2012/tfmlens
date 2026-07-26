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
_WORKER_CONFIG = {
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
        RebalanceFeatureDistribution(**_WORKER_CONFIG["RebalanceFeatureDistribution"]),
        CategoricalFeatureEncoder(**_WORKER_CONFIG["CategoricalFeatureEncoder"]),
        FeatureShuffler(**_WORKER_CONFIG["FeatureShuffler"]),
    ]
    for step in steps:
        x, cat = step.fit_transform(x, cat, seed, y=y)

    eval_pos = len(y_train)
    return x[:eval_pos], x[eval_pos:]


def tabicl_preprocess(X_train, y_train, X_test, categorical_idx, seed=0):
    """Preprocess a (train, test) table for the frozen TabICL forward.

    Reproduces one clean (no-ensemble) member of TabICL's pipeline:
    ``TransformToNumerical`` (ordinal-encodes categorical columns — detected by
    ``category`` dtype) then ``PreprocessingPipeline`` with the code-default
    ``power`` normalization (standard-scale -> yeo-johnson -> outlier-clip). Fits
    on the train (support) rows, transforms both. Returns ``(X_train_p, X_test_p)``
    float32 arrays ready for ``predict_layers``. The 8-way ensemble (none/power x
    feature shuffles) is deferred — self-repair needs a single clean forward.
    """
    import pandas as pd
    from tabicl._sklearn.preprocessing import PreprocessingPipeline, TransformToNumerical

    x = np.concatenate([np.asarray(X_train), np.asarray(X_test)], axis=0)
    df = pd.DataFrame(x)
    for c in categorical_idx:
        df[c] = df[c].astype("category")  # so TransformToNumerical treats it as categorical
    eval_pos = len(y_train)

    encoder = TransformToNumerical().fit(df.iloc[:eval_pos])
    x_num = np.asarray(encoder.transform(df), dtype=np.float32)

    pipe = PreprocessingPipeline(normalization_method="power", random_state=seed)
    pipe.fit(x_num[:eval_pos])
    x_out = np.asarray(pipe.transform(x_num), dtype=np.float32)
    return x_out[:eval_pos], x_out[eval_pos:]


def mitra_preprocess(X_train, y_train, X_test, categorical_idx, seed=0):
    """Preprocess a (train, test) table for the frozen Mitra forward.

    Thin on purpose — Mitra runs its quantile transform inside the model
    (use_quantile_transformer defaults to False), so here we only:

    - ordinal-encode the categorical columns
    - mean-impute the numeric columns
    - zero out any nan/inf

    Fits on the train rows, transforms both. Returns ``(X_train_p, X_test_p)``
    float32; the model's Tab2DQuantileEmbeddingX does the quantile step.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder

    x = np.concatenate([np.asarray(X_train), np.asarray(X_test)], axis=0).astype(np.float64)
    cat = list(categorical_idx)
    num = [i for i in range(x.shape[1]) if i not in cat]
    eval_pos = len(y_train)

    ct = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
            ("num", SimpleImputer(strategy="mean"), num),
        ]
    )
    ct.fit(x[:eval_pos])
    x = np.asarray(ct.transform(x), dtype=np.float32)
    x[~np.isfinite(x)] = 0.0  # nan/inf -> 0
    return x[:eval_pos], x[eval_pos:]
