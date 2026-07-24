"""On-the-fly synthetic training data from the vendored TabICL prior.

``build_prior`` turns a TrainConfig into an endless stream of ``(X, y, eval_pos)``
macro-batches, adapting PriorDataset's 5-tuple output. The mix_scm selection
probabilities are injected into the fixed hyper-params.
"""

from tfm_lens.finetune.config import TrainConfig
from tfm_lens.vendor.tabicl_prior import PriorDataset
from tfm_lens.vendor.tabicl_prior.prior_config import DEFAULT_FIXED_HP


def build_prior(config: TrainConfig):
    p = config.prior
    fixed_hp = {**DEFAULT_FIXED_HP, "mix_probs": tuple(p.mix_probs)}
    dataset = PriorDataset(
        batch_size=config.prior_batch_size,
        device="cpu",
        scm_fixed_hp=fixed_hp,
        prior_type=p.prior_type,
        min_features=p.min_features,
        max_features=p.max_features,
        max_classes=p.max_classes,
        max_seq_len=p.max_seq_len,
        min_train_size=p.min_train_size,
        max_train_size=p.max_train_size,
        batch_size_per_gp=p.batch_size_per_gp,
        log_seq_len=p.log_seq_len,
        n_jobs=p.n_jobs,
    )
    for X, y, _d, _seq_lens, train_sizes in dataset:
        yield X, y, int(train_sizes[0])
