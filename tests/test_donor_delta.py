"""Donor δ capture + role bucketing for resample ablation (#35).

Role integrity is checked by *tagging*: give every donor cell in a bucket a
constant value, then assert each target cell lands on its bucket's tag — proving
context←context / query←query and (4D) label←label / feature←feature.
"""

import numpy as np
import torch

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.donor_delta import build_donor_delta, donor_deltas, layer_deltas
from toys import ToyAdapter3D, ToyAdapter4D

H = ToyAdapter3D.HIDDEN


def _rng():
    return np.random.default_rng(0)


class TestLayerDeltas:
    def test_one_delta_per_layer_3d(self, toy_adapter, toy_input):
        with capture_layers(toy_adapter) as cache:
            toy_adapter.forward_frozen(toy_input, None, 5)
        deltas = layer_deltas(toy_adapter, cache)
        assert len(deltas) == toy_adapter.n_layers
        assert all(d.shape == toy_input.shape for d in deltas)  # [b, seq, hidden]

    def test_delta_reconstructs_output_4d(self, toy_adapter_4d, toy_input_4d):
        # δ = residual_of(out) − residual_of(in) → residual_of(in) + δ == residual_of(out).
        with capture_layers(toy_adapter_4d) as cache:
            toy_adapter_4d.forward_frozen(toy_input_4d, None, 5)
        deltas = layer_deltas(toy_adapter_4d, cache)
        for m, d in enumerate(deltas):
            r_in = toy_adapter_4d.residual_of(cache[m])
            r_out = toy_adapter_4d.residual_of(cache[m + 1])
            torch.testing.assert_close(r_in + d, r_out)

    def test_double_stream_delta_is_a_pair(self, toy_adapter_double_stream, toy_input):
        with capture_layers(toy_adapter_double_stream) as cache:
            toy_adapter_double_stream.forward_frozen(toy_input, None, eval_pos=3)
        deltas = layer_deltas(toy_adapter_double_stream, cache)
        assert len(deltas) == toy_adapter_double_stream.n_layers
        assert all(isinstance(d, tuple) and len(d) == 2 for d in deltas)


class TestBuildDonorDelta3D:
    def test_row_role_match(self):
        # target: 3 context + 2 query; donor: context δ tagged +1, query δ tagged −1.
        target = torch.zeros(1, 5, H)
        donor = torch.empty(1, 4, H)
        donor[:, :2] = 1.0  # 2 context rows
        donor[:, 2:] = -1.0  # 2 query rows
        out = build_donor_delta(
            target, donor, eval_pos_t=3, eval_pos_d=2, label_index=None, rng=_rng()
        )
        assert out.shape == (1, 5, H)
        torch.testing.assert_close(out[:, :3], torch.ones(1, 3, H))  # context ← +1
        torch.testing.assert_close(out[:, 3:], -torch.ones(1, 2, H))  # query ← −1

    def test_reproducible_with_seed(self):
        target, donor = torch.zeros(1, 5, H), torch.randn(1, 4, H)
        kw = dict(eval_pos_t=3, eval_pos_d=2, label_index=None)
        a = build_donor_delta(target, donor, rng=np.random.default_rng(7), **kw)
        b = build_donor_delta(target, donor, rng=np.random.default_rng(7), **kw)
        torch.testing.assert_close(a, b)


class TestBuildDonorDelta4D:
    def test_four_bucket_match(self):
        # tokens: label at −1 (index T-1), features = the rest. Tag all four buckets.
        T = ToyAdapter4D.TOKENS
        target = torch.zeros(1, 4, T, H)  # 2 context + 2 query rows (eval_pos_t=2)
        donor = torch.empty(1, 3, T, H)  # 1 context + 2 query rows (eval_pos_d=1)
        donor[:, :1, -1, :] = 10.0  # context · label
        donor[:, 1:, -1, :] = 20.0  # query   · label
        donor[:, :1, :-1, :] = 1.0  # context · feature
        donor[:, 1:, :-1, :] = 2.0  # query   · feature
        out = build_donor_delta(
            target, donor, eval_pos_t=2, eval_pos_d=1, label_index=-1, rng=_rng()
        )
        assert out.shape == (1, 4, T, H)
        assert torch.all(out[:, :2, -1, :] == 10.0)  # context label ← context label
        assert torch.all(out[:, 2:, -1, :] == 20.0)  # query   label ← query   label
        assert torch.all(out[:, :2, :-1, :] == 1.0)  # context feats ← context feats
        assert torch.all(out[:, 2:, :-1, :] == 2.0)  # query   feats ← query   feats

    def test_label_never_drawn_from_features(self):
        # the hard constraint: a feature δ must never land on the label token.
        T = ToyAdapter4D.TOKENS
        target = torch.zeros(1, 4, T, H)
        donor = torch.zeros(1, 4, T, H)
        donor[..., -1, :] = 99.0  # label column only
        out = build_donor_delta(
            target, donor, eval_pos_t=2, eval_pos_d=2, label_index=-1, rng=_rng()
        )
        assert torch.all(out[..., -1, :] == 99.0)  # label ← label (99)
        assert torch.all(out[..., :-1, :] == 0.0)  # features never see the 99 label tag

    def test_differing_token_counts(self):
        # regression (smoke): donor narrower than target on the token axis (fewer
        # features). Resolve label/feature indices per block — never index the donor
        # with the target's token indices.
        target = torch.zeros(1, 4, 8, H)  # target: 7 feature tokens + 1 label (idx -1)
        donor = torch.zeros(1, 3, 4, H)  # donor: only 3 feature tokens + 1 label
        donor[..., -1, :] = 5.0  # label column
        donor[..., :-1, :] = 1.0  # feature columns
        out = build_donor_delta(
            target, donor, eval_pos_t=2, eval_pos_d=1, label_index=-1, rng=_rng()
        )
        assert out.shape == (1, 4, 8, H)  # keeps the *target* token count
        assert torch.all(out[..., -1, :] == 5.0)  # label ← donor label
        assert torch.all(out[..., :-1, :] == 1.0)  # all 7 target feats ← donor's 3 feats

    def test_donor_wider_than_target(self):
        # the other direction: donor has more feature tokens than the target.
        target = torch.zeros(1, 2, 3, H)  # 2 feature tokens + 1 label
        donor = torch.full((1, 2, 9, H), 1.0)  # 8 feature tokens + 1 label
        donor[..., -1, :] = 7.0
        out = build_donor_delta(
            target, donor, eval_pos_t=1, eval_pos_d=1, label_index=-1, rng=_rng()
        )
        assert out.shape == (1, 2, 3, H)
        assert torch.all(out[..., -1, :] == 7.0) and torch.all(out[..., :-1, :] == 1.0)


class TestBuildDonorDeltaDoubleStream:
    def test_3d_two_streams(self):
        target = (torch.zeros(1, 3, H), torch.zeros(1, 2, H))
        donor = (torch.ones(1, 5, H), -torch.ones(1, 4, H))  # support=+1, query=−1
        out = build_donor_delta(
            target, donor, eval_pos_t=0, eval_pos_d=0, label_index=None, rng=_rng()
        )
        assert isinstance(out, tuple) and len(out) == 2
        torch.testing.assert_close(out[0], torch.ones(1, 3, H))  # support (context)
        torch.testing.assert_close(out[1], -torch.ones(1, 2, H))  # query

    def test_4d_two_streams_with_label_token(self):
        # Mitra-like: double stream AND a token axis, label at index 0.
        T = 3
        target = (torch.zeros(1, 2, T, H), torch.zeros(1, 2, T, H))
        sup, qry = torch.empty(1, 4, T, H), torch.empty(1, 4, T, H)
        sup[:, :, 0, :], sup[:, :, 1:, :] = 10.0, 1.0  # support: label / feature
        qry[:, :, 0, :], qry[:, :, 1:, :] = 20.0, 2.0  # query:   label / feature
        out = build_donor_delta(
            (target[0], target[1]),
            (sup, qry),
            eval_pos_t=0,
            eval_pos_d=0,
            label_index=0,
            rng=_rng(),
        )
        assert torch.all(out[0][:, :, 0, :] == 10.0) and torch.all(out[0][:, :, 1:, :] == 1.0)
        assert torch.all(out[1][:, :, 0, :] == 20.0) and torch.all(out[1][:, :, 1:, :] == 2.0)


class TestDonorDeltasIntegration:
    def test_donor_deltas_runs_forward_and_captures(self, toy_adapter_4d, toy_input_4d):
        deltas = donor_deltas(toy_adapter_4d, toy_input_4d, None, eval_pos=5)
        assert len(deltas) == toy_adapter_4d.n_layers
        assert all(d.shape == toy_input_4d.shape for d in deltas)
