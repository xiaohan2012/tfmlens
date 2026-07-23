---
title: Layer Ablation (Skip / Repeat / Swap)
tags: [interpretability, mechanistic-interpretability, intervention]
sources: [balef2026onelayer]
updated: 2026-05-28
---

# Layer Ablation

Causal interventions on the forward pass to probe what each layer contributes. Three canonical variants, each testing a different property.

## The three interventions

| Intervention | Operation | Tests |
|---|---|---|
| **Skip** $\ell$ | $h_{\ell+1} = h_{\ell-1}$ (residual passes through, layer removed) | **Necessity** — does downstream performance drop? |
| **Repeat** $\ell$ | $h_{\ell+1} = \text{Layer}_\ell(\text{Layer}_\ell(h_{\ell-1}))$ | **Reusability** — does the layer map its output back to a valid input distribution? (loop-body test) |
| **Swap** $\ell, \ell+1$ | apply $\ell+1$ before $\ell$ | **Position-specificity** — is the layer order-bound to its place in the stack? |

## Why these three

- **Skip** alone tells you whether the layer matters at all — but is ambiguous (the layer could be unnecessary, *or* downstream layers could compensate). See [self-repair](self-repair.md) to disambiguate.
- **Repeat** tests whether a layer's transformation is **iteratively applicable** — direct empirical evidence for looped / Universal-Transformer-style architectures.
- **Swap** tests whether layers act as a **sequence** vs. an **unordered set**. LLMs are often surprisingly swap-robust; TFMs are not.

## Caveats

- Single-layer interventions only — multi-layer ablations can compound or cancel.
- "Skip OK" needs [self-repair](self-repair.md) to distinguish *pure redundancy* from *active recovery*.

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — applied across 6 TFMs. Findings: early layers necessary, middle/late skippable; repeats help LIMIX-16M and TabPFN v1 (motivating [looped-transformer-tfm](looped-transformer-tfm.md)); swaps universally hurt, more than in LLMs.
- [ferrando2024primer](ferrando2024primer.md) — places skip/zero-style ablation inside the broader activation-patching family (resample / mean / zero / noise).
- [bilos2026mechanistic](bilos2026mechanistic.md) — per-block knockouts show one dominant early "coordinate-setting" block per model (block 0 / ColEmb-2) while nearly every other block costs <3pp; uniform-attention interventions causally test each readout.
- [gupta2026tabpfnheads](gupta2026tabpfnheads.md) — applies ablation at *attention-head* granularity in TabPFN-2.5; shows selectivity (low attention entropy) is necessary but not sufficient for causal necessity.

## Related Concepts

- [self-repair](self-repair.md) — needed to disentangle skip results.
- [looped-transformer-tfm](looped-transformer-tfm.md) — the repeat intervention motivates the design.
