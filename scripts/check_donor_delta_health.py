"""Donor-δ health check for resample ablation (#35).

Question: is resample **on-manifold + norm-preserving**? Zero ablation is the
"broken" reference. All three checks are read off the *decoded position* (label
token 4D / row vector 3D), over the test rows:

- **③ residual-norm preservation (money plot):** ``r(m) = ‖resid_ablated‖/‖resid_clean‖``.
  resample ≈ 1 (keeps the norm) · zero dips (drops it → the LayerNorm artifact).
- **① δ-norm ratio:** ``‖donor_δ‖ / ‖native_δ‖`` — want ≈ 1 (else rescale donor δ).
- **② direction cosine:** ``cos(donor_δ, native_δ)`` vs within-table native pairs —
  cross-table should look like "another normal δ", not orthogonal.

Cheap: the post-ablation residual *at* layer m needs only layer m's clean input +
the intervention (upstream is frozen), so one clean target forward + one forward
per donor suffices — no per-layer re-forward, no decoders.

    uv run --group eval --group viz python scripts/check_donor_delta_health.py \
        --models limix_2m mitra tabicl_v2 --tasks 363621,363671,363696
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.donor_delta import build_donor_delta, donor_deltas
from tfm_lens.evaluation.datasets import TABARENA_BINARY_TASK_IDS, load_tabarena_task
from tfm_lens.evaluation.preprocess import (
    limix_preprocess,
    mitra_preprocess,
    tabfm_preprocess,
    tabicl_preprocess,
)
from tfm_lens.finetune.__main__ import build_adapter

SEED = 0
PREPROCESS = {
    "limix_2m": limix_preprocess,
    "tabicl_v2": tabicl_preprocess,
    "mitra": mitra_preprocess,
    "tabfm": tabfm_preprocess,
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2"])
    p.add_argument("--tasks", default=None, help="comma-separated task ids; default all 15")
    p.add_argument("--n-donors", type=int, default=8)
    p.add_argument("--subsample-train", type=int, default=500)
    p.add_argument("--subsample-test", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def _subsample(X, y, n):
    if n <= 0 or n >= len(X):
        return X, y
    idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def _load(task_id, preprocess, args):
    X_train, y_train, X_test, y_test, cat_idx = load_tabarena_task(task_id)
    X_train, y_train = _subsample(X_train, y_train, args.subsample_train)
    X_test, y_test = _subsample(X_test, y_test, args.subsample_test)
    classes = np.unique(y_train)
    y_train = np.searchsorted(classes, y_train)
    Xtr_p, Xte_p = preprocess(X_train, y_train, X_test, cat_idx)
    return torch.tensor(Xtr_p), torch.tensor(y_train).float(), torch.tensor(Xte_p)


def _clone(c):
    if isinstance(c, tuple):
        return tuple(t.clone() if torch.is_tensor(t) else t for t in c)
    return c.clone()


def _forward_cache(adapter, Xtr, ytr, Xte, device):
    """Clean per-layer capture cache + eval_pos for one table."""
    eval_pos = Xtr.shape[0]
    X = torch.cat([Xtr, Xte], dim=0).unsqueeze(0).to(device)
    y = ytr.unsqueeze(0).to(device)
    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        return [_clone(c) for c in cache], eval_pos


def _add(a, b):
    if isinstance(a, tuple):
        return tuple(x + y for x, y in zip(a, b, strict=True))
    return a + b


def _sub(a, b):
    if isinstance(a, tuple):
        return tuple(x - y for x, y in zip(a, b, strict=True))
    return a - b


def _decoded(residual, eval_pos, label_index):
    """The decoded position over test rows -> ``[N, hidden]``: query stream / test
    rows, then the label token (4D). This is where the norm/cosine are measured."""
    h = residual[1] if isinstance(residual, tuple) else residual[:, eval_pos:]
    if label_index is not None:
        h = h[..., label_index, :]
    return h.reshape(-1, h.shape[-1])


def _norm(residual, eval_pos, label_index):
    return torch.linalg.norm(_decoded(residual, eval_pos, label_index), dim=-1)  # [N]


def _median(t):
    return float(torch.median(t).item())


def _model_stats(adapter, tables, target_idxs, args):
    """Per-layer ③ (residual-norm) + ② (cosine) for one model.

    ③ is aggregated **per table** (the 8 donors are averaged within a table, per the
    N=8 design), and every table's per-layer value is kept — so the plot can draw a
    median line + IQR/min-max band showing cross-dataset spread. zero is
    donor-independent → one value per table. ② stays a mean over table x donor.
    """
    device = args.device
    li = adapter.label_token_index
    n_layers = adapter.n_layers
    zero_tbl = [[] for _ in range(n_layers)]  # zero_tbl[m] = one ratio per table
    res_tbl = [[] for _ in range(n_layers)]  # res_tbl[m]  = donor-mean ratio per table
    nr, cosA, cosB = ([[] for _ in range(n_layers)] for _ in range(3))

    for ti in target_idxs:
        Xtr, ytr, Xte = tables[ti]
        cache, ep_t = _forward_cache(adapter, Xtr, ytr, Xte, device)
        r_in = [adapter.residual_of(cache[m]) for m in range(n_layers)]
        r_out = [adapter.residual_of(cache[m + 1]) for m in range(n_layers)]
        native = [_sub(r_out[m], r_in[m]) for m in range(n_layers)]
        clean_n = [_norm(r_out[m], ep_t, li) for m in range(n_layers)]

        donors = [j for j in range(len(tables)) if j != ti]
        picks = np.random.default_rng(SEED).permutation(donors)[: args.n_donors]
        res_per_donor = []  # [n_donors, n_layers] for this table
        for di, dj in enumerate(picks):
            Xd_tr, yd_tr, Xd_te = tables[dj]
            d_deltas = donor_deltas(
                adapter,
                torch.cat([Xd_tr, Xd_te]).unsqueeze(0).to(device),
                yd_tr.unsqueeze(0).to(device),
                Xd_tr.shape[0],
            )
            draw = np.random.default_rng(SEED + 1 + di)
            res_m = []
            for m in range(n_layers):
                dd = build_donor_delta(r_in[m], d_deltas[m], ep_t, Xd_tr.shape[0], li, draw)
                res_m.append(_median(_norm(_add(r_in[m], dd), ep_t, li) / clean_n[m]))
                nat_dec, don_dec = _decoded(native[m], ep_t, li), _decoded(dd, ep_t, li)
                nr[m].append(
                    _median(torch.linalg.norm(don_dec, dim=-1) / torch.linalg.norm(nat_dec, dim=-1))
                )
                cosB[m].append(_median(torch.cosine_similarity(don_dec, nat_dec, dim=-1)))
                perm = torch.randperm(nat_dec.shape[0])
                cosA[m].append(_median(torch.cosine_similarity(nat_dec, nat_dec[perm], dim=-1)))
            res_per_donor.append(res_m)
        res_mean = np.mean(res_per_donor, axis=0)  # donor-mean per layer, this table
        for m in range(n_layers):
            zero_tbl[m].append(_median(_norm(r_in[m], ep_t, li) / clean_n[m]))  # donor-independent
            res_tbl[m].append(float(res_mean[m]))

    agg = lambda rows: [float(np.mean(r)) for r in rows]  # noqa: E731
    return {
        "ratio_zero_tbl": zero_tbl,  # [n_layers][n_tables] — for median + band
        "ratio_resample_tbl": res_tbl,
        "ratio_zero": [float(np.median(r)) for r in zero_tbl],  # summary median line
        "ratio_resample": [float(np.median(r)) for r in res_tbl],
        "cos_within": agg(cosA),
        "cos_cross": agg(cosB),
    }


def _plot(stats, out):
    models = list(stats)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), squeeze=False)
    for ax, name in zip(axes.ravel(), models, strict=True):
        s = stats[name]
        n = len(s["ratio_zero"])
        x = np.arange(n) / max(n - 1, 1)
        ax.axhline(1.0, ls=":", color="0.5", lw=1)
        ax.plot(x, s["ratio_zero"], "--o", color="tab:red", ms=3, label="zero (skip)")
        ax.plot(x, s["ratio_resample"], "-o", color="tab:blue", ms=3, label="resample")
        ax.set_title(f"{name}  ·  ‖resid_ablated‖ / ‖resid_clean‖")
        ax.set_xlabel("relative depth (0=first · 1=last)")
        ax.set_ylabel("residual-norm ratio @ decoded pos")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "V1 · residual-norm preservation — resample keeps the norm, zero drops it",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [int(t) for t in args.tasks.split(",")] if args.tasks else TABARENA_BINARY_TASK_IDS

    stats = {}
    for model in args.models:
        adapter = build_adapter(model, None, args.device)
        tables, ok = [], []
        for tid in task_ids:
            try:
                tables.append(_load(tid, PREPROCESS[model], args))
                ok.append(tid)
            except Exception as exc:
                print(f"{model} load {tid}: FAILED {type(exc).__name__}: {exc}", flush=True)
        s = _model_stats(adapter, tables, list(range(len(tables))), args)
        stats[model] = s
        # ②/③ one-line summary (median over layers / tables)
        cc, cw = np.median(s["cos_cross"]), np.median(s["cos_within"])
        zmin = min(s["ratio_zero"])
        print(
            f"{model}: ② cos cross={cc:.2f} vs within={cw:.2f} | "
            f"③ resample median-line min={min(s['ratio_resample']):.2f} "
            f"zero median-line min={zmin:.2f}",
            flush=True,
        )
        del adapter
        if args.device == "cuda":
            torch.cuda.empty_cache()

    (args.out_dir / "v1_stats.json").write_text(json.dumps(stats, indent=2))
    _plot(stats, args.out_dir / "v1_residual_norm.png")


if __name__ == "__main__":
    main()
