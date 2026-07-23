---
title: Self-Repair
tags: [interpretability, mechanistic-interpretability, intervention]
sources: [balef2026onelayer]
updated: 2026-05-28
---

# Self-Repair

When a layer is removed, downstream layers can sometimes **reconstruct** its computation. First observed in LLMs (McGrath et al. 2023); shown to transfer to TFMs by [Balef et al. 2026](balef2026onelayer.md). Distinguishes **active redundancy** (depth as safety net) from **dead layers** (depth as pure overhead).

## Why it matters

[Layer ablation](layer-ablation.md) measures only *final-layer* accuracy. "Skip OK" is ambiguous between:

- **Pure redundancy.** Layer was unnecessary; nothing to recover; no dip ever happened.
- **Self-repair.** Layer was necessary; downstream layers reconstructed the missing computation.

Both yield identical final-layer accuracy. Distinguishing them is required evidence for *iterative inference* and for looped / Universal-Transformer designs (each pass must be able to correct previous-pass errors).

## Technique — lens-after-skip trajectory

1. Pick a target layer $\ell^*$, apply residual identity $h_{\ell^*+1} = h_{\ell^*-1}$.
2. At every subsequent layer $\ell \geq \ell^*+1$, emit a real prediction via the [tabular logit lens](tabular-logit-lens.md) → per-layer AUC.
3. Compare each post-skip trace against the no-skip baseline; repeat for every $\ell^*$.

## Reading the trace

- **Dip then recovery toward baseline** → self-repair. Depth where AUC re-joins baseline = *repair distance*.
- **No dip** → pure redundancy (the layer wasn't contributing).
- **Dip with no recovery** → layer uniquely necessary (no safety net).

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — middle/late TFM skips show clear self-repair (especially TabPFN v2); first-layer skips never recover.
- [ferrando2024primer](ferrando2024primer.md) — flags self-repair as a known confound of causal interventions in LLMs (McGrath et al. 2023; Rushing 2024).
- [bilos2026mechanistic](bilos2026mechanistic.md) — after knocking out a coordinate-setting block, a probe *retrained* on the post-knockout activations still recovers labels while the *frozen* probe collapses — class information survives but in a frame downstream layers cannot read.

## Related Concepts

- [layer-ablation](layer-ablation.md) — generates the skip whose effect self-repair measures.
- [tabular-logit-lens](tabular-logit-lens.md) — the instrument that produces the per-layer AUC trajectory.
