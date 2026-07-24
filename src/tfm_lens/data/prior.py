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
    # mix_probs is routed into scm_fixed_hp; every other PriorConfig field maps
    # 1:1 onto a PriorDataset kwarg, so forward them straight through.
    fixed_hp = {**DEFAULT_FIXED_HP, "mix_probs": tuple(p.mix_probs)}
    dataset = PriorDataset(
        batch_size=config.prior_batch_size,
        device="cpu",
        scm_fixed_hp=fixed_hp,
        **p.model_dump(exclude={"mix_probs"}),
    )
    for X, y, _d, _seq_lens, train_sizes in dataset:
        yield X, y, int(train_sizes[0])
