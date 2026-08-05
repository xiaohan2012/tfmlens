# de-te-figures — moved out of HEAD

The PNGs that used to live here were `git rm`'d. **Nothing is broken.**

- The PR #45 comments embed them via **SHA-pinned** `raw.githubusercontent.com/.../<commit>/analysis/de-te-figures/<file>.png`.
- A pinned URL serves the blob **as of that commit**, so removing the file in a later commit does not affect it.
- Those commits stay reachable via `refs/pull/45/head` (GitHub keeps PR refs permanently).

Only branch-name URLs or a history rewrite would break them.

> `git rm` does **not** shrink the repo — the blobs remain in history. The point is a clean HEAD: no stale copy of `out/` that nobody regenerates.

## Where they are pinned

| figure | commit | used in |
|---|---|---|
| `fig1_scatter_margin.png` | `40a6af6` | [results comment](https://github.com/xiaohan2012/tfmlens/pull/45#issuecomment-5192047221) — E1 |
| `fig2_perrow_cehist_margin.png` | `40a6af6` | results comment — E1 (details) |
| `hydra_ref_scatter.png`, `hydra_ref_regression.png` | `40a6af6` | results comment — Hydra paper reference (screenshots, not generated) |
| `layer_law_agg_margin.png` | `9521d59` | results comment — E2, Hydra Fig-4d analog |
| `apex_scatter_agg_margin.png` | `9521d59` | results comment — E2, Hydra Fig-4b analog |
| `dip_recover_fig8.png` | `dbc1629` | [method-note comment](https://github.com/xiaohan2012/tfmlens/pull/45#issuecomment-5195523559) |
| `fig3_hydra_layer_margin.png` | `40a6af6` | *unused* — superseded strawman (single hand-picked layer), replaced by the Fig-4d curve |

Retrieve any of them with `git show <commit>:analysis/de-te-figures/<file>.png > /tmp/<file>.png`.

## Regenerating

Input = the final per-row sweep (`out/perrow/de_<model>.json`, 15 tasks × 1000/500, resample).

```bash
# E1 — aggregate DE–TE scatter
uv run --group viz python scripts/plot_de_te_scatter.py \
    --in-dir out/perrow --coord margin --out out/fig1_scatter_margin.png

# E1 — per-row CE histogram (needs --per-row json)
uv run --group viz python scripts/plot_de_te_perrow.py \
    --in-dir out/perrow --coord margin --hist-only --out out/fig2_perrow_cehist_margin.png

# E2 — per-layer CE~DE law across depth (Hydra Fig-4d)
uv run --group viz python scripts/plot_compensation_fit.py \
    --in-dir out/perrow --coord margin --agg --out out/layer_law_agg_margin.png

# E2 — DE vs CE at the apex layer (Hydra Fig-4b)
uv run --group viz python scripts/plot_compensation_fit.py \
    --in-dir out/perrow --coord margin --agg --apex-scatter --out out/apex_scatter_agg_margin.png
```

- `dip_recover_fig8.png` = a copy of `out/balef_exp6_combined_fig8.png` (exp6 tuned-decoder trajectory), from an uncommitted local plot script.
- `hydra_ref_*.png` are screenshots from McGrath et al. 2023 — not reproducible from this repo.
