# tfm_lens

A clean, extensible toolkit for the **tabular logit lens** — reproducing
experiments 4/5/6 (classification only) of Balef et al., *Is One Layer Enough?*
(ICML 2026).

The three experiments share one primitive:

```
frozen forward → capture per-layer residual stream → [optional intervention] → decode to prediction
```

- **Exp4** — train per-layer decoders (the logit-lens tool itself; the one asset that costs compute).
- **Exp5** — looping / repeating layers, quantified with the logit lens.
- **Exp6** — self-repair: skip a layer, quantify how downstream layers compensate.

Model-specific code lives only in `adapters/`; capture and interventions use
plain PyTorch forward hooks (no model-source surgery, no third-party lens libs).

See `DESIGN.md` for the full design and build plan.

## Development

```bash
uv sync                 # create .venv and install deps + dev group
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
pre-commit install      # enable git hooks
```
