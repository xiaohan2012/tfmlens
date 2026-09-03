# Figures for the write-up

Scripts that produce the figures of
[*Do tabular foundation models repair themselves?*](https://xiaohan2012.github.io/articles/tfm-self-repair/).

Each reads the sweep outputs already in `out/` and writes PNGs to
`out/article-figures/`. Run them from anywhere:

    uv run --group viz python analysis/article-figures/<script>.py

| script | figure in the article | reads |
|---|---|---|
| `make_exp6_transition.py` | Fig 3 — layer skipping, schematic and LimiX-2M | `out/self_repair_full.json` |
| `make_ablation_figs.py` | Fig 7, 8, 9, 17 and the numbers of Table 4 | `out/rr_v1_all/v1_stats.json`, `out/v2_{model}_{zero,resample}.json` |
| `make_regions.py` | Fig 10 — the five regions of the DE–TE plane | nothing; it is a schematic |
| `make_de_te_scatter.py` | Fig 11 — DE against TE, four models | `out/de_{model}.json` |
| `make_compensation_fit.py` | Fig 12 — every layer at its (slope, R²) | `out/de_{model}.json` |

The remaining figures are not produced here: Fig 1 and 2 are taken from
McGrath et al.; Fig 5 and 6 come from the Exp6 sweep figures of #39; Fig 13, 14,
15 and 16 are existing outputs in `out/`.

To update the article, regenerate and copy across:

    cp out/article-figures/*.png ../xiaohan2012.github.io/assets/img/tfm-self-repair/

The repo README's hero figure is not produced here — it is the single-panel
form of Fig 11, rendered straight from the library script:

    uv run --group viz python scripts/plot_de_te_scatter.py \
        --models limix_2m --coord margin --out docs/figures/de-te-limix.png
