# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored preprocessing subset from tabfm/src/classifier_and_regressor.py.

Byte-for-byte (Apache-2.0) extract of the numeric-normalization classes TabFM's
shared preprocessing applies before the frozen forward: CustomStandardScaler ->
(optional) normalizer -> OutlierRemover. Only the classes reproducing ONE clean
ensemble member are kept; the EnsembleGenerator / FeatureShuffler /
TransformToNumerical / sklearn estimator wrappers are not vendored.
"""

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted, validate_data

class OutlierRemover(TransformerMixin, BaseEstimator):
  """Clips extreme values based on training data distribution.

  This implementation uses a two-stage Z-score based approach to identify and
  clip outliers:
  1. First stage: Identify values beyond z standard deviations and mark as
  missing
  2. Second stage: Recompute statistics without outliers for more robust bounds
  3. Final stage: Apply log-based clipping to maintain data distribution

  Attributes:
    means_: Per-feature means after outlier removal.
    stds_: Per-feature standard deviations after outlier removal.
    lower_bounds_: Per-feature clipping lower bounds.
    upper_bounds_: Per-feature clipping upper bounds.
  """

  means_: np.ndarray
  stds_: np.ndarray
  lower_bounds_: np.ndarray
  upper_bounds_: np.ndarray

  def __init__(self, threshold: float = 4.0):
    """Initialises the remover.

    Args:
      threshold: Values beyond this many standard deviations are outliers.
    """
    self.threshold = threshold

  def fit(self, X: Any, y: Any = None) -> "OutlierRemover":
    """Learn clipping bounds from training data.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)

    # First stage: Identify outliers using initial statistics
    self.means_ = np.nanmean(X, axis=0)
    self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

    # Ensure standard deviations are not zero
    self.stds_ = np.maximum(self.stds_, 1e-6)

    # Create a clean copy with outliers replaced by NaN
    X_clean = X.copy()
    lower_bounds = self.means_ - self.threshold * self.stds_
    upper_bounds = self.means_ + self.threshold * self.stds_

    # Create masks for values outside bounds
    lower_mask = X < lower_bounds[np.newaxis, :]
    upper_mask = X > upper_bounds[np.newaxis, :]
    outlier_mask = np.logical_or(lower_mask, upper_mask)

    # Set outliers to NaN
    X_clean[outlier_mask] = np.nan

    # Second stage: Recompute statistics without outliers
    self.means_ = np.nanmean(X_clean, axis=0)
    self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

    # Ensure standard deviations are not zero
    self.stds_ = np.maximum(self.stds_, 1e-6)

    # Compute final bounds
    self.lower_bounds_ = self.means_ - self.threshold * self.stds_
    self.upper_bounds_ = self.means_ + self.threshold * self.stds_

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Clip values based on learned bounds with log-based adjustments.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Clipped array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
    X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)
    return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
  """Standard scaling with clipping.

  This scaler computes the mean and standard deviation of the training data,
  adds a small epsilon to the standard deviation to avoid division by zero,
  and clips the transformed values to a reasonable range.

  Attributes:
    mean_ : ndarray of shape (n_features,)
      The mean value for each feature in the training set.

    scale_ : ndarray of shape (n_features,)
      The standard deviation for each feature in the training set with epsilon
      added.
  """

  mean_: np.ndarray
  scale_: np.ndarray

  def __init__(
      self,
      clip_min: float = -100,
      clip_max: float = 100,
      epsilon: float = 1e-6,
  ):
    """Initialises the scaler.

    Args:
      clip_min: Lower bound for clipping scaled values.
      clip_max: Upper bound for clipping scaled values.
      epsilon: Small constant added to the standard deviation to avoid
        division by zero.
    """
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.epsilon = epsilon

  def fit(self, X: Any, y: Any = None) -> "CustomStandardScaler":
    """Compute the mean and std to be used for scaling.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)
    self.mean_ = np.mean(X, axis=0)
    self.scale_ = np.std(X, axis=0) + self.epsilon
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Scaled and clipped array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    X_scaled = (X - self.mean_) / self.scale_
    return np.clip(X_scaled, self.clip_min, self.clip_max)


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
  """Quantile transformer adapted for tabular deep learning models.

  This implementation is based on research from the RTDL group and adds noise to
  training
  data before applying quantile transformation, improving robustness and
  generalization.
  It also dynamically adjusts the number of quantiles based on data size.

  Attributes:
    normalizer_: The fitted ``QuantileTransformer``.

  Notes:
    Adapted from
    https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
    following
    https://github.com/dholzmueller/pytabkit/blob/949bf81e3964f65a33dd2c252c3713c239c17b2d/pytabkit/models/utils.py#L431
  """

  normalizer_: QuantileTransformer

  def __init__(
      self,
      noise: float = 1e-3,
      n_quantiles: int = 1000,
      subsample: int = 1_000_000_000,
      output_distribution: str = "normal",
      random_state: Optional[int] = None,
  ):
    """Initialises the transformer.

    Args:
      noise: Relative magnitude of Gaussian noise to add. Set to 0 to disable.
      n_quantiles: Maximum number of quantiles. Actual number is determined
        dynamically as ``min(n_samples // 30, n_quantiles)``, with a floor of
        10.
      subsample: Maximum samples used to estimate quantiles.
      output_distribution: Target marginal distribution (``"uniform"`` or
        ``"normal"``).
      random_state: Seed for reproducibility.
    """
    self.noise = noise
    self.n_quantiles = n_quantiles
    self.subsample = subsample
    self.output_distribution = output_distribution
    self.random_state = random_state

  def fit(self, X: Any, y: Any = None) -> "RTDLQuantileTransformer":
    """Fit the quantile transformer to training data with optional noise.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    # Calculate the number of quantiles based on data size
    n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

    # Initialize QuantileTransformer
    normalizer = QuantileTransformer(
        output_distribution=self.output_distribution,
        n_quantiles=n_quantiles,
        subsample=self.subsample,
        random_state=self.random_state,
    )

    # Add noise if required
    X_modified = self._add_noise(X) if self.noise > 0 else X

    # Fit the normalizer
    normalizer.fit(X_modified)

    # Show that it's fitted
    self.normalizer_ = normalizer

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transform data using the fitted quantile transformer.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Transformed array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    return self.normalizer_.transform(X)

  def _add_noise(self, X: np.ndarray) -> np.ndarray:
    """Add noise to the input data proportional to feature standard deviations.

    The noise magnitude is controlled by the 'noise' parameter and is scaled
    inversely to the standard deviation of each feature to ensure
    consistent noise levels across features of different scales.

    Args:
      X: Array of shape (n_samples, n_features).

    Returns:
      Noisy array of shape (n_samples, n_features).
      The input data with added Gaussian noise.
    """
    stds = np.std(X, axis=0, keepdims=True)
    noise_std = self.noise / np.maximum(stds, self.noise)
    rng = np.random.default_rng(self.random_state)
    return X + noise_std * rng.standard_normal(X.shape)


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
  """Preprocessing pipeline combining scaling, normalization, and outlier handling.

  Attributes:
    standard_scaler_: Fitted ``CustomStandardScaler``.
    normalizer_: Fitted normalization transformer (or ``None``).
    outlier_remover_: Fitted ``OutlierRemover``.
    X_transformed_: Cached transformed training data.
    X_min_: Per-feature minimum of scaled data (used for clipping at transform).
    X_max_: Per-feature maximum of scaled data (used for clipping at transform).
  """

  n_features_in_: int
  standard_scaler_: Optional[CustomStandardScaler]
  normalizer_: Optional[Any]
  outlier_remover_: Optional[OutlierRemover]
  X_transformed_: np.ndarray
  X_min_: np.ndarray
  X_max_: np.ndarray

  def __init__(
      self,
      normalization_method: str = "none",
      outlier_threshold: float = 4.0,
      random_state: Optional[int] = None,
  ):
    """Initialises the pipeline.

    Args:
      normalization_method: Normalization strategy: ``"none"``, ``"power"``,
        ``"quantile"``, ``"quantile_rtdl"``, or ``"robust"``.
      outlier_threshold: Z-score threshold for outlier detection.
      random_state: Seed for reproducible normalization.
    """
    self.normalization_method = normalization_method
    self.outlier_threshold = outlier_threshold
    self.random_state = random_state

  def fit(self, X: Any, y: Any = None) -> "PreprocessingPipeline":
    """Fit the preprocessing pipeline.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X, ensure_min_features=0)
    # If there are no features, there's nothing to preprocess.
    if self.n_features_in_ == 0:
      self.standard_scaler_ = None
      self.normalizer_ = None
      self.outlier_remover_ = None
      self.X_transformed_ = X
      return self
    # 1. Apply standard scaling
    self.standard_scaler_ = CustomStandardScaler()
    X_scaled = self.standard_scaler_.fit_transform(X)

    # 2. Apply normalization
    if self.normalization_method != "none":
      if self.normalization_method == "power":
        self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
      elif self.normalization_method == "quantile":
        self.normalizer_ = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
      elif self.normalization_method == "quantile_rtdl":
        self.normalizer_ = Pipeline([
            (
                "quantile_rtdl",
                RTDLQuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                ),
            ),
            ("std", StandardScaler()),
        ])
      elif self.normalization_method == "robust":
        self.normalizer_ = RobustScaler(unit_variance=True)
      else:
        raise ValueError(
            f"Unknown normalization method: {self.normalization_method}"
        )
      self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
      self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
      X_normalized = self.normalizer_.fit_transform(X_scaled)
    else:
      self.normalizer_ = None
      X_normalized = X_scaled

    # 3. Handle outliers
    self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
    self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Apply the preprocessing pipeline.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Preprocessed array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False, copy=True, ensure_min_features=0)
    if self.n_features_in_ == 0:
      return X
    if self.standard_scaler_ is not None:
      X = self.standard_scaler_.transform(X)
    if self.normalizer_ is not None:
      try:
        # this can fail in rare cases if there is an outlier in X that was not present in fit()
        X = self.normalizer_.transform(X)
      except ValueError:
        # clip values to train min/max
        X = np.clip(X, self.X_min_, self.X_max_)
        X = self.normalizer_.transform(X)
    if self.outlier_remover_ is not None:
      X = self.outlier_remover_.transform(X)
    return X


