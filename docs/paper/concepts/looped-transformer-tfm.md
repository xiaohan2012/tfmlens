---
title: Looped Transformer for TFMs
tags: [tabular, architecture, efficiency, looped-transformer, tabpfn]
sources: [balef2026onelayer]
updated: 2026-05-25
---

# Looped Transformer for Tabular Foundation Models

Applying a *single* transformer block recurrently in place of a deep stack — a parameter-efficient alternative motivated by the depth-redundancy observed in TFMs ([Balef et al. 2026](balef2026onelayer.md)).

## Background

- **Idea:** train one block, apply it `L` times — trade parameters for compute, induce iterative refinement.
- **Weight sharing is the defining property.** All `L` passes reuse the *same* parameters (vs. a standard stack, which has `L` independent weight sets). Hence ~1/L the parameters at equal depth-of-compute.
- **Lineage:** Universal Transformer (Dehghani 2018); scaling work by Gong 2025, Zhu 2025, McLeish 2025.

## TFM-specific motivation

[Balef et al. 2026](balef2026onelayer.md) finds depth in TFMs mostly buys iterative refinement (depth-redundancy, self-repair, block structure in embedding similarity); repeating a layer even slightly improves LimiX-16M and [TabPFN v1](hollmann2023tabpfnv1.md).

## nanoTabPFN experiment

Three models pretrained on TabICL priors from the open-source [nanoTabPFN](https://github.com/automl-private/nanoTabPFN) codebase ([TabPFN v2](hollmann2025tabpfnv2.md)-style architecture):

| Model                  | Layers | Params  | Compute |
| ---------------------- | ------ | ------- | ------- |
| `nanoTabPFN_{6l}`      | 6      | full    | 6x      |
| `nanoTabPFN_{1l}`      | 1      | ~17%    | 1x      |
| `nanoTabPFN_{looped}`  | 1 (×6) | ~17%    | 6x      |

On PMLBmini and TabArena:

- `nanoTabPFN_{looped}` ≈ `nanoTabPFN_{6l}` on AUC.
- `nanoTabPFN_{1l}` clearly worse.
- Gains aren't from parameter count (matched to 1l) — they're from the iterative compute.

![Looped 1-layer nanoTabPFN matches the 6-layer baseline; the standalone 1-layer model is clearly worse.](assets/looped-transformer-tfm-nanotabpfn.png)

## Practical advantages

- ~5× parameter reduction at equal performance.
- **Any-time predictions:** loop count is a tunable inference budget; no per-layer decoder pretraining needed (cf. [tabular logit lens](tabular-logit-lens.md)).
- Adaptive computation: harder tasks could trigger more loops.

## Caveats

- Demonstrated only at the nanoTabPFN scale; no direct evidence that it scales to TabPFN(2.5) or LimiX-16M sizes.
- Looped models still trail SOTA like TabPFN(2.5) in absolute performance — it's a proof of concept for the design principle, not a replacement.

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — first TFM application; nanoTabPFN-scale evidence that depth ≈ recurrence.

## Related Concepts

- [tfm-inference-stages](tfm-inference-stages.md) — the redundancy and self-repair findings motivate looping.
- [tabular-logit-lens](tabular-logit-lens.md) — looping sidesteps the need for per-layer decoders.
